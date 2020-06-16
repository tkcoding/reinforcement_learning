# Create environment models for use . WW21.1 is creating similar environment for use.
import numpy as np
from models import Machine, Jobs
import warnings
import random
from collections import defaultdict, OrderedDict, namedtuple
import itertools
import pandas as pd
# Design reward in a way that is cater for CQT and nThroughput
# environment changes
# Dual process need to be coded
# Action only includes releasing from the staging area.
#Edited at abc

# Introduce machine status that will be up or down at a certain percentage.
class Environment(object):
    def __init__(self, config):
        self.config = config
        # Initializing alll the process time needed for CVD and CDO
        self.process_completion_cvd = np.zeros((self.config['NUM_MACHINES_CVD'], self.config['NUM_JOBS']))
        self.process_completion_cdo = np.zeros((self.config['NUM_MACHINES_CDO'], self.config['NUM_JOBS']))
        self.df_log = pd.read_csv("log_data_per_day.csv")


    def reset(self):
        NUM_MACHINES_CVD = self.config['NUM_MACHINES_CVD']
        NUM_MACHINES_CDO = self.config['NUM_MACHINES_CDO']
        NUM_JOBS = self.config['NUM_JOBS']
        self.machines_cvd = [Machine(str(i),process='CVD') for i in range(NUM_MACHINES_CVD)]
        self.machines_cdo = [Machine(str(i),process='CDO') for i in range(NUM_MACHINES_CDO)]
        self.jobs = [Jobs(i) for i in range(NUM_JOBS)]
        self.miss_cqt = []
        self.NOW = 0
        self.jobs_processed_CVD = np.zeros((self.config['NUM_MACHINES_CVD'],self.config['NUM_JOBS']), dtype=np.int8)
        self.jobs_processed_CDO = np.zeros((self.config['NUM_MACHINES_CDO'],self.config['NUM_JOBS']), dtype=np.int8)
        self.process_completion_cvd = np.zeros((self.config['NUM_MACHINES_CVD'], self.config['NUM_JOBS']))
        self.process_completion_cdo = np.zeros((self.config['NUM_MACHINES_CDO'], self.config['NUM_JOBS']))
        self.lot_release = 0
        self.time_release = self.df_log['Time step'].values
        self.job_num = 0
        return self.returnObs()

    def getProcessTime(self, macID, jID , process):
        # import pdb; pdb.set_trace()
        # This part is to implement ASOM job and machine processing time
        # Taking from current data table run time
        if process == 'CVD':
            if self.jobs_processed_CVD[macID, jID] == 0:
                t_time = 55 + random.gauss(5, 3) # Include some variation
                self.jobs_processed_CVD[macID, jID] = t_time
                return self.jobs_processed_CVD[macID, jID]
        else: # CDO
            if self.jobs_processed_CDO[macID, jID] == 0:
                t_time = 50 + random.gauss(5, 3)
                self.jobs_processed_CDO[macID, jID] = t_time
                return self.jobs_processed_CDO[macID, jID]


    def returnObs(self):
        # Return observation based on lots to entity ratio
        # For example : CVD : 5/14 , CDO 4/4 (This matrix includes available entity and lots release)
        obs = {}
        # How do you start with randomly dispatch lots .

        # number of lots , ratio , number of queue lot
        _active_lots_cvd = 0
        _active_lots_cdo = 0
        _instaging = 0
        ########## Update entities availability from time to time. #############
        # if self.NOW == "":
            # change numer of machine available to mimic the environment of down
            # entity
        ##################
        if self.NOW in self.time_release:
            if not self.jobs[self.job_num].inStaging:
                self.jobs[self.job_num].inStaging = True
                self.lot_release = self.lot_release + 1
                self.job_num += 1


        for each_job in range(self.config['NUM_JOBS']):
            if self.jobs[each_job].inStaging:
                _instaging += 1

        # Calculate how many lots in the loop ?
        for i in range(self.config['NUM_JOBS']):
            # Each lots active check.
            if self.jobs[i].jobBusy_p1:
                _active_lots_cvd += 1
            elif self.jobs[i].jobBusy_p2 or self.jobs[i].cvd_completion:
                _active_lots_cdo += 1
            # Adding another elif for number of queue lots
            # Adding another elif for available lot at staging area. (This is to replace with dynamic lot arrival)
            else:
                pass
        # print("DEBUG on cvd : {} and cdo : {}".format())
        total_queue = 0
        for i in range(self.config['NUM_JOBS']):
            if self.jobs[i].inQueue:
                total_queue += 1
        ratio_lots_cvd = _active_lots_cvd/self.config['NUM_MACHINES_CVD']
        ratio_lots_cdo = _active_lots_cdo/self.config['NUM_MACHINES_CDO']

        # obs['lot_ratio'] = tuple([ratio_lots_cvd,ratio_lots_cdo])
        # obs['lot_ratio'] = np.digitize(_active_lots_cvd,bins)
        bins = np.linspace(0,10,5)
        obs['queue_lot'] = np.digitize(total_queue,bins)
        obs['lot_number'] = np.digitize(_instaging,bins)
        obs['action_num'] = _instaging
        return obs

    def returnQuadInfo(self, done=False):
        # Return all the parameter that environment need.
        # Observation, rewards , done signal and info

        obs = self.returnObs()
        if self.NOW == 0:
            reward = 0
        else:
            reward = self.calcMetrics()
        # print("Missing Lot : ",reward)
        info = "nothing" # Redundant now
        return (obs, reward, done, info)

    def calcMetrics(self):
        # Calculate the throughput of how many jobs has been done for the day.
        # Drum beat calculation
        cqt_lot = 0
        totalJobsDone = np.sum(self.process_completion_cdo)
        for each_job in range(self.config['NUM_JOBS']):
            if self.jobs[each_job].process_queue_time > 60 and each_job not in self.miss_cqt:
                self.miss_cqt.append(each_job)
                cqt_lot += 1
        return cqt_lot

    # Specially made for environment machine retrieval information
    # In the current mode should be always getting fully occupied machine.
    # For each of the machines need to have two separate process
    def getEmptyMachines(self,process):
        if process == "CVD":
            return [i for i in range(self.config['NUM_MACHINES_CVD']) if self.machines_cvd[i].machineBusy is False]
        else:
            return [i for i in range(self.config['NUM_MACHINES_CDO']) if self.machines_cdo[i].machineBusy is False]

    def getBusyMachines(self,process):
        if process == 'CVD':
            return [i for i in range(self.config['NUM_MACHINES_CVD']) if self.machines_cvd[i].machineBusy is True]
        else:
            return [i for i in range(self.config['NUM_MACHINES_CDO']) if self.machines_cdo[i].machineBusy is True]


    def getEmptyJobs(self,process):
        if process == 'CVD':
            return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy_p1 is False and self.jobs[i].next_process == 'CVD' and self.jobs[i].inStaging]
        else:
            return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy_p2 is False and self.jobs[i].next_process == 'CDO' and not self.jobs[i].cdo_completion]

    def getBusyJobs(self,process):
        if process == 'CVD':
            return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy_p1 is True]
        else:
            return [i for i in range(self.config['NUM_JOBS']) if self.jobs[i].jobBusy_p2 is True]


    def checkCompletion(self):
        # Calculate total job to mark the completion
        totalJobsDone = int(np.sum(self.process_completion_cdo))
        toDo = int(self.config['NUM_JOBS'])
        if totalJobsDone == toDo:
            return True
        else:
            return False

    def queue_time_addition(self):
        queue_for_cdo = self.getEmptyJobs('CDO')
        for each_lot in queue_for_cdo:
            self.jobs[each_lot].process_queue_time = self.jobs[each_lot].process_queue_time + 1
        return


    def step(self, action):
        quad = self.takeAction(action)
        return quad

    def scheduleJob(self, machineID, jobID, process , time_):
        # Scedule the job to work on machine (Depending on the flow)
        if process == "CVD":
            self.jobs[jobID].getProcessed(process)
            p_time = self.getProcessTime(machineID, jobID, process)
            self.machines_cvd[machineID].processJob(jobID, time_, p_time)
        else:
            self.jobs[jobID].getProcessed(process)
            p_time = self.getProcessTime(machineID, jobID, process)
            self.machines_cdo[machineID].processJob(jobID, time_, p_time)
        self.config['SCHEDULE_ACTION'].append({'MachineID':machineID, 'JobId':jobID, 'scheduledAt':time_, 'ProcessTime':p_time})
        return

    def takeAction(self, action):
        # Always give the action for the longest lot stay inside the list
        # Release or not release.
        # Every single time step the environment might have changes.

        if self.checkCompletion() is True:
            print("Simulation over")
            return self.returnQuadInfo(done=True)
        self.queue_time_addition() # Adding into queue time counting
        if action:
            job_id = self.getEmptyJobs('CVD') # Getting CVD job.
            macID = self.getEmptyMachines('CVD')
            process = 'CVD' #TODO: Initial process to CVD , there is 8 actions due to it consolidate
            if action > len(job_id):
                action = len(job_id)
            for count in range(action): # Count of how many times we can dispatch
                # How can i arrange the lot so that it can be schedule out .
                _macID = self.getEmptyMachines('CVD')
                if len(_macID) < 1:
                    pass
                else:
                    # print('Time now : {} , dispatching : {}'.format(self.NOW,self.jobs[job_id[count]].jobID))
                    self.scheduleJob(self.machines_cvd[_macID[0]].machineID, self.jobs[job_id[count]].jobID,process,self.NOW)



        if self.getEmptyJobs('CDO'):
            print("Empty Jobs time : {} , list  {} ".format(self.NOW,self.getEmptyJobs('CDO')))
            for eachJob in self.getEmptyJobs('CDO'):
                process = 'CDO'

                if self.getEmptyMachines('CDO'):
                    # print("Empty Machine : ",self.getEmptyMachines('CDO'))
                    # print('CDO Time now : {} , dispatching : {}'.format(self.NOW,self.jobs[eachJob].jobID))
                    self.scheduleJob(self.getEmptyMachines('CDO')[0], self.jobs[eachJob].jobID,process,self.NOW)
                else:
                    pass

        # Getting status from machine and release job
        busy_cvd = self.getBusyMachines('CVD')
        busy_cdo = self.getBusyMachines('CDO')

        for i, bM in enumerate(busy_cvd):
            jOT = self.machines_cvd[bM].jobOverTime
            if int(self.NOW) == int(jOT):
                jobID = self.machines_cvd[bM].onJob
                self.release(bM, jobID, 'CVD',self.NOW) # Release from staging area
                self.jobs[jobID].inQueue = True

        for i, bM in enumerate(busy_cdo):
            jOT = self.machines_cdo[bM].jobOverTime
            if int(self.NOW) == int(jOT):
                jobID = self.machines_cdo[bM].onJob
                self.release(bM, jobID,'CDO', self.NOW) # Release from staging area
                self.jobs[jobID].inQueue = False

        # if self.checkCompletion() is True:
        #     print("Simulation over")
        #     return self.returnQuadInfo(done=True)
            # how to update observation needed information?

            # below not needed..
            # emptyMac = self.getEmptyMachines()
            # since there are empty mac request for review
            # if (len(emptyMac) != 0):
            #     self.NOW += 1
            #     return self.returnQuadInfo(done=False)

        self.NOW += 1 # Basic time step for current simulation use.


        #if self.NOW > self.config['SIMULATION_TIME']:
        if self.NOW > self.config['SIMULATION_TIME']:
            return self.returnQuadInfo(done=True)
        return self.returnQuadInfo(done=False)


    def release(self, machineID, jobID, process ,time_):

        if process == "CVD":
            assert int(self.machines_cvd[machineID].jobOverTime) == time_
            self.machines_cvd[machineID].releaseMachine()
            self.jobs[jobID].releaseJob(process)
            self.process_completion_cvd[machineID, jobID] = 1
            # p_time = self.getProcessTime(machineID,jobID,process)
        else:
            assert int(self.machines_cdo[machineID].jobOverTime) == time_
            self.machines_cdo[machineID].releaseMachine()
            self.jobs[jobID].releaseJob(process)
            self.process_completion_cdo[machineID, jobID] = 1
            # p_time = self.getProcessTime(machineID, jobID,process)
        self.config['RELEASE_ACTION'].append({'MachineID':machineID, 'JobId':jobID, 'releasedAt':time_,'process':process})
        return

    def getValidJobs(self,machineID,process):
        # Govern the movement of lot to enter the flow of CVD -> CDO
        # 1. Only output lots that is valid for the specific process.
        # 2. lot that still in the list and waiting to be released
        # 3. lot completed at CVD ONLY able to go to CDO
        # 4. No reversal process from CDO -> CVD

        if process == 'CVD':
            if self.machines_cvd[machineID].machineBusy is True:
                return -1
            else:
                valid_jobs = []
                empJobs = self.getEmptyJobs(process)
                for i,j in enumerate(empJobs):
                    j_done_len = self.jobs[j].machineVisited
                    if j_done_len < self.config['NUM_MACHINES_CVD']:
                        toDo = self.config['ORDER_OF_PROCESSING_CVD'][j][j_done_len]
                        if toDo == machineID and self.jobs[j].inStaging:
                            valid_jobs.append(j)
                if len(valid_jobs) == 0:
                    valid_jobs = -1
        else:
            if self.machines_cdo[machineID].machineBusy is True:
                return -1
            else:
                valid_jobs = []
                # first get empty jobs
                empJobs = self.getEmptyJobs(process)
                for i,j in enumerate(empJobs):
                    j_done_len = self.jobs[j].machineVisited
                    if j_done_len < self.config['NUM_MACHINES_CDO']:
                        toDo = self.config['ORDER_OF_PROCESSING_CDO'][j][j_done_len]
                        if toDo == machineID:
                            valid_jobs.append(j)
                if len(valid_jobs) == 0:
                    valid_jobs = -1
        return valid_jobs


    def generatePossibleAction(self, obs):
        # Generating possible action randomly.
        # Only release process at CVD.
        # CDO shouldn't be included.
        #

        possibleAction = []     # list of possible action to take
        state_info = {}     # state information


        # obs.items() -> arrival lots ,
        # it will be return ((3,0.5,10) , 3)
        for key,value in obs.items():
            if key == 'lot_ratio':
                state_info['lot_ratio'] = value
            elif key == 'action_num':
                possibleAction = [x for x in range(value+1)]
                # print("Possible Action : ",possibleAction)
            elif key == 'queue_lot':
                state_info['queue_lot'] = value
            elif key == 'lot_number':
                state_info['staging_lot'] = value
            # What should be the state information here ?
            # New observation at this point ? Increase in the number of entity_lot_ration ?
        _active_lots_cvd = 0
        _active_lots_cdo = 0
        ########## Update entities availability from time to time. #############
        # if self.NOW == "":
            # change numer of machine available to mimic the environment of down
            # entity
        ##################
        # Calculate how many lots in the loop ?
        # for i in range(self.config['NUM_JOBS']):
        #     # Each lots active check.
        #     if self.jobs[i].jobBusy_p1:
        #         _active_lots_cvd =+ 1
        #     elif self.jobs[i].jobBusy_p2 or self.jobs[i].cvd_completion:
        #         _active_lots_cdo =+ 1
        #     else:
        #         pass

        # ratio_lots_cvd = _active_lots_cvd/self.config['NUM_MACHINES_CVD']
        # ratio_lots_cdo = _active_lots_cdo/self.config['NUM_MACHINES_CDO']
        # state_info['CVD_ratio'] = ratio_lots_cvd
        # state_info['CDO_ratio'] = ratio_lots_cdo
        '''
        for machine, status in obs.items():
            process = machine[1] # First item in the tuple is this
            machine = machine[0]
            # Need to follow sequence for those who is
            if status == -1: # If status available (-1)
                valJob = self.getValidJobs(machine,process)
                print("Val job :",valJob)
                # import pdb; pdb.set_trace()
                if valJob == -1:
                    possibleAction.append({tuple([machine,process]):-1})
                else:
                    if not isinstance(valJob, list):
                        valJob = [valJob]
                    valJob += [-1]
                    possibleAction.append({tuple([machine,process]):valJob}) # Create possible combination
                if process == "CVD":
                    state_info[machine,process] = tuple(self.machines_cvd[machine].jobsDone)
                else:
                    state_info[machine,process] = tuple(self.machines_cdo[machine].jobsDone)

            else:
                possibleAction.append({tuple([machine,process]):-1}) # If occupied then append machine:-1
                if process == "CVD":
                    state_info[machine,process] = tuple(self.machines_cvd[machine].jobsDone + [status])
                else:
                    state_info[machine,process] = tuple(self.machines_cdo[machine].jobsDone + [status])

        print("SInfo: ",state_info)
        print("Paction: ",possibleAction)
        # End of computing possible actions and state info
        # import pdb; pdb.set_trace()
        permuteAct = []
        machines = []
        for act in possibleAction:
            t_ = list(act.values())[0]
            if not isinstance(t_, list):
                t_ = [t_]
            permuteAct.append(t_)
            machines.append(list(act.keys())[0])

        totalAct = itertools.product(*permuteAct)

        actions = []

        for act in totalAct:
            temp = {}
            act_ = list(act)
            act_ = [a_ for a_ in act_ if a_!=-1]
            # import pdb; pdb.set_trace()
            if len(set(list(act_))) == len(list(act_)):
                for i,a_ in enumerate(act):
                    temp[machines[i]] = a_
                actions.append(temp)
            else:
                continue
        # How do we deal with this .
        tempAct = []


        for a in actions:
            # Need to fit in process information
            tempAct.append(tuple(OrderedDict(sorted(a.items())).values()))

        if tempAct == []:
            tempAct = [tuple([-1]*self.config['NUM_MACHINES'])]
        '''
        # Sorted dictionary does it have the wrong effect on the observation ?
        state_info = tuple(OrderedDict(state_info.items()).values())

        return state_info, possibleAction,value

    def completedLotNumber(self):
        """
            Input : None
            Output: Return number of lot completed CDO (last operation)
            Lot number completed check for end of the reward
        """
        completed_lot = 0
        instaging = 0
        for each_job in range(self.config['NUM_JOBS']):
            if self.jobs[each_job].cdo_completion:
                completed_lot += 1
        return completed_lot
