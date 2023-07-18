
import numpy as np
import pandas as pd

import misc

class LogReader():

    def __init__(self, filepath:str):

        self.fix = []

        self.load(filepath)

        return
    
    # -------------------------------------------------------------------------
    
    def load(self, filepath:str):
        
        fix = []
        raw = []
        with open(filepath) as file:
            i = 0
            for line in file:
                line = file.readline().strip().split(",")

                match line[0]:
                    case "Raw":
                        mdict = misc.getRawDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Raw'")
                        else:
                            raw.append(mdict)
                    case "Fix":
                        mdict = misc.getFixDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Fix'")
                        else:
                            fix.append(mdict)
                    
                i += 1
        
        # Convert to dataframes
        self.fix = pd.DataFrame(fix)
        self.raw = pd.DataFrame(raw)
        
        print(self.raw)

        return
    
    # -------------------------------------------------------------------------
    
# =====================================================================================================================

class PosReader():

    def __init__(self, filepath:str):

        self.pos = []

        self.load(filepath)

        return
    
    def load(self, filepath:str):
        
        pos = []
        with open(filepath) as file:
            i = 0
            for line in file:
                if line[0] == '%':
                    continue

                mdict = misc.getPosDictionnary(line)
                if mdict is None:
                    print(f"Warning: Line {i} skipped with invalid values for 'Pos'")
                else:
                    pos.append(mdict)
            
                i += 1
        
        # Convert to dataframes
        self.pos = pd.DataFrame(pos)
        
        # print(self.pos)

        return
    
# =====================================================================================================================

if __name__ == "__main__":
    
    filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
    log = LogReader(filepath)

    filepath = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
    log = PosReader(filepath)