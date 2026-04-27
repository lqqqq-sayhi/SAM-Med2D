import pandas as pd
import os

def generate_sam_med2d_summaries():
    base_dir = "/mnt/hdd2/task2/sam-med2d/predict"
    instrument_classes = set(range(1, 26))
    organ_classes = {26, 27, 28}
    
    for fold in range(5):
        input_filename = f"fold{fold}_predict_per_image_detailed_metrics.csv"
        output_filename = f"fold{fold}_organ_instrument.csv"
        input_path = os.path.join(base_dir, input_filename)
        output_path = os.path.join(base_dir, output_filename)
        
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue
            
        print(f"Processing {input_path}...")
        df = pd.read_csv(input_path)
        
        summary_rows = []
        for label, target_classes in [("Instrument", instrument_classes), ("Organ", organ_classes)]:
            sub_df = df[df['class'].isin(target_classes)]
            if not sub_df.empty:
                summary_rows.append({
                    "Group": label,
                    "Mean Dice": sub_df['Dice'].mean(),
                    "Mean IoU": sub_df['IOU'].mean(),
                    "Mean HD95": sub_df['HD95'].mean()
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_sam_med2d_summaries()
