from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import main
if __name__=='__main__':main(forced_modes=['experiment2_gas_temperature'],default_output=ROOT/'experiment2_gas_temperature/runs/independent_v1')
