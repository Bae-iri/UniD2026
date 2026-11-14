from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run import main
if __name__=='__main__':main(forced_modes=['experiment1_voltage'],default_output=ROOT/'experiment1_voltage/runs/independent_v1')
