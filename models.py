import numpy as np
import os
import random

"""
Author : Teck Keat , Yeow
Revision V0.1
- Machines and jobs are created here as a common module
- Machine ID is purely as indexes for use , machine name is a unique identifier
- Machine Busy is state that currently coded for simplicity purpose without considering actual state yet.
    - TODO : Edit to machine stat instead
- Current machine have two state

"""
class Machine(object):
    # Entity will consists of up and downstate too.
    # First step is to assume all machine up at all time.
    def __init__(self, name,process):
        self.machineID = int(name)
        self.machineName = "Machine_"+str(name)
        self.machineBusy = False
        self.jobOverTime = 0
        self.now = 0
        self.jobsDone = []
        self.process = process
        #TODO :  self.status # cater for change of state of down and up. Can add additional status

    def processJob(self, jobID, time, pTime):
        # check if machine is busy or not
        assert self.machineBusy == False
        self.onJob = jobID
        self.now = time
        self.processTime = pTime
        # import pdb; pdb.set_trace()
        self.jobOverTime = self.now + self.processTime
        self.machineBusy = True
        return

    def releaseMachine(self):
        # check if currently in use
        assert self.machineBusy == True
        self.jobsDone.append(self.onJob)
        self.machineBusy = False
        return

"""
ASOM job attribute
Pilot on CVD -> CDO Loop
"""
class Jobs(object):
    def __init__(self, name):
        self.jobID = name
        self.jobName = "Job_"+str(name)
        self.jobBusy_p1 = False
        self.jobBusy_p2 = False
        self.processDetails = []
        self.now = 0
        self.cvd_completion = False
        self.cdo_completion = False
        self.machineVisited = 0
        self.inQueue = False
        self.process_queue_time = 0
        self.next_process = 'CVD'
        self.noOfProcess = 0
        self.inStaging = False

        # self.priority = 0

    def getProcessed(self,process):
        # print("{} requested".format(self.jobName))
        if process == 'CVD':
            assert self.jobBusy_p1 == False
            self.jobBusy_p1 = True
            self.inStaging = False
        else:
            assert self.jobBusy_p2 == False
            self.jobBusy_p2 = True
        return

    def releaseJob(self,process):
        # print("{} released".format(self.jobName))
        if process == 'CVD':
            assert self.jobBusy_p1 == True
            self.machineVisited += 1
            self.jobBusy_p1 = False
            self.cvd_completion = True
            self.next_process = 'CDO'
            self.inQueue = False
        else:
            assert self.jobBusy_p2 == True
            self.machineVisited += 1
            self.jobBusy_p2 = False
            self.cdo_completion = True
        return
