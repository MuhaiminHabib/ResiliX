import subprocess
import sys
import time

def run_script(script_name, description):
    print("=" * 85)
    print(f"RUNNING: {description} ({script_name})")
    print("=" * 85)
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, script_name], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"[SUCCESS] {script_name} completed in {time.time() - start_time:.2f}s\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed while running {script_name}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

def main():
    print("*" * 85)
    print(" RESILIX: UNIFIED MASTER EXPERIMENT PIPELINE FOR DISSERTATION REPRODUCIBILITY")
    print("*" * 85 + "\n")
    
    # 1. Run Syn3 Cross-Domain Evaluation
    run_script("eval_syn3.py", "Cross-Domain Syn3 Tree-Grids Evaluation")
    
    # 2. Run Elliptic Bitcoin Real-World AML Evaluation
    run_script("eval_elliptic_resilix.py", "Real-World Elliptic Bitcoin AML Evaluation")
    
    # 3. Run IBM AMLSim Account Transfer Evaluation
    run_script("eval_ibm_aml_resilix.py", "IBM AMLSim HI-Small Account Transfer Evaluation")
    
    # 4. Run Hyperparameter Sensitivity Analysis
    run_script("eval_sensitivity.py", "Hyperparameter Sensitivity Sweep & Stability Plot")
    
    # 5. Run Computational Latency Profiling
    run_script("eval_latency.py", "Computational Overhead & Latency Profiling")
    
    # 6. Run Cross-Attack Generalization Evaluation
    run_script("eval_cross_attacks.py", "Cross-Attack Generalization (Edge, Feature, Gradient)")
    
    print("*" * 85)
    print(" ALL DISSERTATION EXPERIMENTS AND PROFILES COMPLETED SUCCESSFULLY!")
    print("*" * 85)

if __name__ == '__main__':
    main()
