
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

        misc.resetGlobal()
        
        fix = []
        raw = []
        health = []
        motion = []
        env = []

        mode = 'logger'
        if "mimir" in filepath:
            mode = 'mimir'
        elif "old" in filepath:
            mode = 'old'

        with open(filepath) as file:
            i = 0
            for line in file:
                line = file.readline().strip().split(",")

                match line[0]:
                    case "Raw":
                        mdict = misc.getRawDictionnary(line, mode)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Raw'")
                        else:
                            raw.append(mdict)
                    
                    case "Fix":
                        mdict = misc.getFixDictionnary(line, mode)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'Fix'")
                        else:
                            fix.append(mdict)

                    case "ECG":
                        mdict = misc.getHealthDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'ECG'")
                        else:
                            health.append(mdict)

                    case "PPG":
                        mdict = misc.getHealthDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'PPG'")
                        else:
                            health.append(mdict)

                    case "ACC":
                        mdict = misc.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'ACC'")
                        else:
                            motion.append(mdict)
                    
                    case "GYR":
                        mdict = misc.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'GYR'")
                        else:
                            motion.append(mdict)
                    
                    case "MAG":
                        mdict = misc.getMotionDictionnary(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'MAG'")
                        else:
                            motion.append(mdict)

                    case "PSR":
                        mdict = misc.getEnvironmentDict(line)
                        if mdict is None:
                            print(f"Warning: Line {i} skipped with invalid values for 'MAG'")
                        else:
                            env.append(mdict)
                    
                i += 1
        
        # Convert to dataframes
        self.fix = pd.DataFrame(fix)
        self.raw = pd.DataFrame(raw)
        self.health = pd.DataFrame(health)
        self.motion = pd.DataFrame(motion)
        self.env = pd.DataFrame(env)

        print(self.motion)

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
    
    #filepath = "./.data/gnss_log_2023_04_14_15_23_32.txt"
    #filepath = "./.data/log_old_20230414152332.txt"
    filepath = "./.data/log_mimir_20230715122058.txt"
    log = LogReader(filepath)

    filepath = "./.data/NMND18410025C_2023-04-14_13-03-45.pos"
    log = PosReader(filepath)