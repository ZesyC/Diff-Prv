import subprocess
import sys
import os
import json
import argparse
import pandas as pd

def run_cfm(dataset, epochs, gpu):
    print(f"\n" + "="*60)
    print(f"Training CFM OPTIMIZED on dataset: {dataset.upper()} ({epochs} epochs)")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, "Main.py")
    
    cmd = [
        sys.executable, "-u", main_script,
        "--data", dataset,
        "--epoch", str(epochs),
        "--gpu", gpu
    ]
    
    # Setup params based on original papers for specific datasets
    if dataset == 'tiktok':
        cmd += ["--reg", "1e-4", "--ssl_reg", "1e-2", "--trans", "1", "--e_loss", "0.1", "--cl_method", "1"]
    elif dataset == 'baby':
        cmd += ["--reg", "1e-5", "--ssl_reg", "1e-1", "--keepRate", "1", "--e_loss", "0.01"]
    elif dataset == 'sports':
        cmd += ["--reg", "1e-5", "--ssl_reg", "1e-2", "--keepRate", "0.5", "--e_loss", "0.01"]
        
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding='utf-8', 
        cwd=script_dir
    )
    
    for line in process.stdout:
        try:
            print(line, end='', flush=True)
        except Exception:
            pass
        
    process.wait()
    print(f"Finished CFM OPTIMIZED on {dataset.upper()}! (Exit code: {process.returncode})")
    
    res_path = os.path.join(script_dir, f"results_cfm_optimized_{dataset}.json")
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Result file not found at {res_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Chạy CFM optimized trên các tập dữ liệu')
    parser.add_argument('--epoch', default=2, type=int, help='Số epoch huấn luyện mặc định là 2 để sinh bảng nhanh')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    args_comp = parser.parse_args()

    epochs = args_comp.epoch
    gpu = args_comp.gpu

    datasets = ['tiktok', 'baby', 'sports']
    
    all_results = []
    
    for dataset in datasets:
        res = run_cfm(dataset, epochs, gpu)
        if res:
            all_results.append({
                'Dataset': dataset.upper(),
                'Recall@20': f"{res['recall']:.6f}",
                'NDCG@20': f"{res['ndcg']:.6f}",
                'Precision@20': f"{res['precision']:.6f}"
            })

    print("\n\n" + "="*80)
    print("CFM OPTIMIZED RESULTS ACROSS ALL DATASETS".center(80))
    print("="*80 + "\n")
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        print(df_all.to_string(index=False))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, f"benchmark_report_epoch_{epochs}.md")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Báo Cáo CFM Optimized\n\n")
            f.write(f"**Số epoch huấn luyện (Demo):** {epochs}\n\n")
            f.write("| Dataset | Recall@20 | NDCG@20 | Precision@20 |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for row in all_results:
                f.write(f"| {row['Dataset']} | {row['Recall@20']} | {row['NDCG@20']} | {row['Precision@20']} |\n")
            
            f.write("\n\n## Nhận xét\n")
            f.write("- **CFM Optimized**: dùng vector field liên tục, Euler solver và behavior-guided condition cho user-modal preference.\n")
            
        print(f"\nSuccessfully exported full comparison report to '{report_path}'")
    else:
        print("Error: Could not get enough results to build the table.")

if __name__ == '__main__':
    main()
