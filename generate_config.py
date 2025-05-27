import pandas as pd
import numpy as np
import io # Required to read the CSV string directly

# --- Paste your CSV data here as a multi-line string ---
csv_data_string = """Width,Height,BatchSize,Time_s,PeakMemory_GB
256,256,1,0.7328841090202332,0.25295722484588623
256,256,2,1.2517074743906658,0.4759302139282226
256,256,4,2.2163768808046975,0.9465341567993164
256,256,8,4.16298379500707,1.8875741958618164
256,256,12,6.172664801279704,2.829667091369629
256,256,16,7.965257227420807,3.771180152893065
256,256,24,12.01952713727951,5.654206275939941
256,256,32,16.064785480499268,7.537964820861816
512,256,1,2.710390110810598,0.5030789375305176
512,256,2,3.3515923817952475,0.9449243545532228
512,256,4,5.4798977971076965,1.8845224380493164
512,256,8,10.460399230321249,3.763550758361816
512,256,12,14.93995589017868,5.643631935119629
512,256,16,19.762735327084858,7.523133277893066
512,256,24,29.076607803503673,11.28213596343994
512,256,32,38.7266499598821,12.03778886795044
512,512,1,3.3169713020324707,1.0033230781555176
512,512,2,6.3690396547317505,1.8829126358032229
512,512,4,12.280068735281626,3.760499000549317
512,512,8,23.75118225812912,7.515503883361816
512,512,12,35.74086399873098,11.271561622619627
512,512,16,48.208018481731415,12.0229229927063
768,512,1,5.829822421073914,1.5035672187805176
768,512,2,10.941535969575247,2.8209009170532227
768,512,4,21.845810373624165,5.636475563049316
768,512,8,43.396393060684204,11.267457008361816
768,512,12,66.61385701100032,13.5207781791687
768,768,1,6.635613838831584,2.2539334297180176
768,768,2,14.245516876379648,4.227883338928223
768,768,4,29.889632900555927,8.450882911682129
768,768,8,62.224107106526695,13.517321109771729
1024,512,1,7.035386939843495,2.0038113594055176
1024,512,2,15.798851450284323,3.758889198303223
1024,512,4,33.23782515525818,7.512452125549316
1024,512,8,68.95420227448146,12.016252994537354
1024,768,1,15.522455036640167,3.0042996406555176
1024,768,2,33.561785797278084,5.634865760803223
1024,768,4,69.64489376544952,11.264405250549316
1024,1024,1,19.460586587587994,4.004787921905518
1024,1024,2,42.96888949473699,7.510842323303223
1024,1024,4,88.15774180491765,12.011971950531006
1536,768,1,31.690625846385956,4.505032062530518
1536,768,2,67.60027748346329,8.449273109436035
1536,768,4,147.34192349513373,13.513879299163818
1536,1024,1,33.57016692558924,6.005764484405518
1536,1024,2,71.80964181820552,11.262795448303224
1536,1536,1,98.3454093337059,9.007473468780518
1536,1536,2,224.3566182653109,13.51153326034546
2048,1024,1,59.650014440218605,8.006741046905518
2048,1024,2,130.2414309581121,12.010602474212646
2048,1536,1,251.89724715550742,12.008694171905518
2048,2048,1,448.55239059527713,12.009670734405518
"""

def generate_config_from_csv_string(csv_string):
    """
    Reads performance data from a CSV string and generates Python code
    for config variables.
    """
    try:
        # Use io.StringIO to read the string as if it were a file
        df = pd.read_csv(io.StringIO(csv_string))
    except Exception as e:
        print(f"Error reading CSV string: {e}")
        return

    # Use 'Width' as the primary key for image size buckets to align with current env.py
    # The effect of 'Height' will be averaged out for Time_s and PeakMemory_GB
    # if multiple Heights exist for the same Width-BatchSize combination.
    
    unique_img_widths = sorted(df['Width'].unique())
    unique_batch_sizes = sorted(df['BatchSize'].unique())

    print("######################################################################")
    print("# START: Generated Python code for config.py (using benchmark_results_refine.csv)")
    print("######################################################################")
    print("import numpy as np\n")

    print("# --- Dynamically generated from benchmark_results_refine.csv ---")
    print(f"IMG_SIZE_BUCKETS = np.array({unique_img_widths}, dtype=np.float32)")
    print(f"BATCH_SIZE_OPTIONS = np.array({unique_batch_sizes}, dtype=np.int32)")
    print(f"K_DIM_BUCKETS: int = {len(unique_img_widths)}")
    print(f"M_BATCH_SIZE_OPTIONS: int = {len(unique_batch_sizes)}")
    print("# Note: You might need to update N_PRIORITY_WEIGHTS and STRUCTURED_ACTION_FLAT_DIM manually if they changed.")
    print("# Current N_PRIORITY_WEIGHTS is assumed from previous config (e.g., 3).")
    print(f"# STRUCTURED_ACTION_FLAT_DIM: int = K_DIM_BUCKETS + M_BATCH_SIZE_OPTIONS + N_PRIORITY_WEIGHTS # Example, adjust N_PRIORITY_WEIGHTS")


    # Initialize matrices with NaN
    processing_times_matrix = np.full((len(unique_img_widths), len(unique_batch_sizes)), np.nan)
    peak_memory_matrix = np.full((len(unique_img_widths), len(unique_batch_sizes)), np.nan)

    # Group by Width and BatchSize, then average Time_s and PeakMemory_GB
    # This handles multiple Height entries for the same Width-BatchSize.
    grouped = df.groupby(['Width', 'BatchSize'])
    
    for name, group in grouped:
        width, batch_size = name
        avg_time_s = group['Time_s'].mean()
        avg_peak_memory_gb = group['PeakMemory_GB'].mean()
        
        try:
            width_idx = unique_img_widths.index(width)
            batch_idx = unique_batch_sizes.index(batch_size)
            
            processing_times_matrix[width_idx, batch_idx] = avg_time_s
            peak_memory_matrix[width_idx, batch_idx] = avg_peak_memory_gb
        except ValueError:
            print(f"Warning: Index not found for Width {width} or BatchSize {batch_size}. This shouldn't happen if lists are from df uniques.")

    print("\n# PROCESSING_TIMES_BY_INDEX (Time_s from CSV, averaged over Height for same Width-BatchSize)")
    print("# Rows: IMG_SIZE_BUCKETS (based on Width), Columns: BATCH_SIZE_OPTIONS")
    print(f"PROCESSING_TIMES_BY_INDEX = np.array({np.round(processing_times_matrix, 6).tolist()})")
    print("# Ensure NaN values are handled (e.g., replaced with defaults or large numbers if a combination is impossible/missing)")

    print("\n# PEAK_MEMORY_GB_BY_INDEX (PeakMemory_GB from CSV, averaged over Height for same Width-BatchSize)")
    print("# This represents the memory for the *entire batch* of this type.")
    print("# Rows: IMG_SIZE_BUCKETS (based on Width), Columns: BATCH_SIZE_OPTIONS")
    print(f"PEAK_MEMORY_GB_BY_INDEX = np.array({np.round(peak_memory_matrix, 6).tolist()})")
    print("# Ensure NaN values are handled.")
    
    print("\n# --- Other configurations (review and adjust as needed) ---")
    print("# Example: GPU_TOTAL_CAPACITY_GB = 16.0 # Set this to your actual server GPU memory in GB")
    print("# STEP_DURATION_SECONDS: float = 0.5 # Adjust based on your simulation's time scale")

    print("\n######################################################################")
    print("# END: Generated Python code")
    print("######################################################################")
    print("\nIMPORTANT NOTES:")
    print("1. Copy and paste the generated code sections into your `config.py` file.")
    print("2. Manually replace any `np.nan` values in the generated arrays with sensible defaults.")
    print("   - For `PROCESSING_TIMES_BY_INDEX`, `NaN` could mean the combination is not in your CSV.")
    print("     You might replace it with a very large number (e.g., `float('inf')` or `env.max_steps_per_episode * config.STEP_DURATION_SECONDS`)")
    print("     or an extrapolated value if the agent should still be able to select it.")
    print("   - For `PEAK_MEMORY_GB_BY_INDEX`, `NaN` could be replaced with 0, a large number (if it means it won't fit), or an extrapolated value.")
    print("3. Set `GPU_TOTAL_CAPACITY_GB` in your `config.py` to your actual server's GPU memory if you're using the `PEAK_MEMORY_GB_BY_INDEX` approach.")
    print("4. Adjust `STEP_DURATION_SECONDS` in `config.py` to define how real-world seconds from your CSV map to environment steps.")
    print("5. The script averages `Time_s` and `PeakMemory_GB` if multiple `Height` values exist for the same `Width` and `BatchSize`.")
    print("   If `Height` is a critical independent factor for performance that needs to be distinctly modeled,")
    print("   your `IMG_SIZE_BUCKETS` and related logic in `env.py` would need to handle 2D image dimensions (e.g., tuples like (Width, Height)).")

if __name__ == '__main__':
    # generate_config_from_csv_string(csv_data_string)

    try:
        with open("benchmark_results_refine.csv", 'r') as f:
            csv_file_content = f.read()
        generate_config_from_csv_string(csv_file_content)
    except FileNotFoundError:
        print("Error: benchmark_results_refine.csv not found. Please create it with your CSV data.")

