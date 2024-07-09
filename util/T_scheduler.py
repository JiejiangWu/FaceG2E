from util.dreamtime import dreamtime_t
import os
import numpy as np
dreamtime_m1 = 800;dreamtime_m2=500;dreamtime_s1=300;dreamtime_s2=500

class T_scheduler(object):
    def __init__(self,schedule_type,total_optim_steps,m1=dreamtime_m1,m2=dreamtime_m2,s1=dreamtime_s1,s2=dreamtime_s2,max_t_step=999):
        self.schedule_type = schedule_type
        self.N = total_optim_steps
        self.T = max_t_step
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2

        self.compute_max_min_T_step(self.T)
        self.pre_compute_t_table()

    def compute_max_min_T_step(self,num_train_timesteps):
        if self.schedule_type in ['uniform_rand','linear']:
            max_step = int(num_train_timesteps * 0.98)
            min_step = int(num_train_timesteps * 0.02)
        elif self.schedule_type == 'dreamtime':
            max_step = num_train_timesteps
            min_step = 1
        
        self.min_step = min_step
        self.max_step = max_step
        return max_step, min_step

    def compute_t(self,i):
        if self.schedule_type == 'uniform_rand':
            t = None
        if self.schedule_type == 'linear':
            interval = (self.max_step - self.min_step) / self.N            
            t = int(self.max_step - interval * i)
        if self.schedule_type == 'dreamtime':
            t = int(self.t_table[i])
        return t

    def pre_compute_t_table(self):
        os.makedirs('./T_schedule',exist_ok=True)
        if self.schedule_type == 'dreamtime':
            t_table_filename = f'./T_schedule/dreamtime_N_{self.N}_m1_{self.m1}_m2_{self.m2}_s1_{self.s1}_s2_{self.s2}.npy'
            if os.path.exists(t_table_filename):
                self.t_table = np.load(t_table_filename)
            else:
                self.t_table = dreamtime_t(self.m1,self.m2,self.s1,self.s2,self.min_step,self.max_step,self.N)
                np.save(t_table_filename,self.t_table)
