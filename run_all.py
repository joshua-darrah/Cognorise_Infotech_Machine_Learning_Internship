from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
projects=[
('Project 1 — House Price Prediction', ROOT/'Project_1_House_Price_Prediction'/'train_and_analyze.py'),
('Project 3 — Employee Attrition Prediction', ROOT/'Project_3_Employee_Attrition_Prediction'/'train_and_analyze.py'),
('Project 5 — Customer Churn Prediction', ROOT/'Project_5_Customer_Churn_Prediction'/'train_and_analyze.py')]
for name, script in projects:
    print('\n'+'='*72+'\n'+name+'\n'+'='*72)
    result=subprocess.run([sys.executable,str(script)],cwd=str(script.parent))
    if result.returncode != 0:
        raise SystemExit(f'{name} failed with exit code {result.returncode}')
print('\nAll three Cognorise Infotech projects completed successfully.')
