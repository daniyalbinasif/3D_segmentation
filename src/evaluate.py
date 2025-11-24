import os
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
from config import TEST_DIR, CHECKPOINT_DIR, RESULTS_DIR
from utils import find_volumes, compute_metrics_np, show_slice_comparison
from data_loader import preprocess_pair
from model import dice_coef, bce_dice_loss

keras.utils.get_custom_objects().update({
    'dice_coef': dice_coef, 
    'bce_dice_loss': bce_dice_loss
})

def evaluate_on_test(model, test_pairs, experiment_name, results_dir):
    all_results = []
    vis_dir = results_dir / experiment_name / 'predictions'
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {experiment_name} on {len(test_pairs)} test cases")
    
    for i, (img_path, msk_path) in enumerate(tqdm(test_pairs)):
        img_vol, msk_vol = preprocess_pair(img_path, msk_path) 
        pred_vol = model.predict(img_vol[np.newaxis, ...], verbose=0)[0]
        metrics = compute_metrics_np(msk_vol, pred_vol)
        case_name = Path(img_path).stem.replace('_image', '')
        print(f"\n Case {case_name} Metrics ({experiment_name})")
        case_df = pd.Series(metrics, name="Value").to_frame().round(4)
        print(case_df.to_markdown(numalign="left", stralign="left"))
        all_results.append({'case': case_name, 'experiment': experiment_name, **metrics})
        
        show_slice_comparison(
            img_vol.squeeze(-1), 
            msk_vol.squeeze(-1), 
            pred_vol.squeeze(-1), 
            title=f'{case_name} - {experiment_name}',
            results_dir=vis_dir
        )
        
    df_results = pd.DataFrame(all_results)
    mean_metrics = df_results[['Dice', 'Precision', 'Sensitivity', 'HD95']].mean().to_dict()
    return df_results, mean_metrics


def main():
    test_pairs = find_volumes(TEST_DIR)
    if not test_pairs:
        print("No test data found.")
        return

    experiments = {
        'Baseline': CHECKPOINT_DIR / 'Baseline' / 'best_model.h5',
        'Intensity_Augmentation': CHECKPOINT_DIR / 'Intensity_Augmentation' / 'best_model.h5',
        'Elastic_Deformation_Augmentation': CHECKPOINT_DIR / 'Elastic_Deformation_Augmentation' / 'best_model.h5',
    }
    
    all_mean_results = {}
    
    for exp_name, model_path in experiments.items():
        if not model_path.exists():
            print(f"Skipping {exp_name}: Model not found at {model_path}.")
            continue
        print(f"\n{'='*60}\nEvaluating Experiment: {exp_name}\n{'='*60}")

        model = keras.models.load_model(str(model_path))
        df_results, mean_metrics = evaluate_on_test(model, test_pairs, exp_name, RESULTS_DIR)
        all_mean_results[exp_name] = mean_metrics
        df_results.to_csv(RESULTS_DIR / f'{exp_name}_detailed_metrics.csv', index=False)
        
        print(f"\n{exp_name} Mean Metrics")
        print(pd.Series(mean_metrics).round(4).to_dict())

    if all_mean_results:
        print("\n" + "="*60)
        print("FINAL EXPERIMENT SUMMARY (Mean Test Metrics)")
        print("="*60)
        df_summary = pd.DataFrame.from_dict(all_mean_results, orient='index').round(4)
        print(df_summary.to_markdown())
        df_summary.to_csv(RESULTS_DIR / 'summary_comparison.csv')
        
if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main()