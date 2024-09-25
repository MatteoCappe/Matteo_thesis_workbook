from bimvee.importIitYarp import importIitYarp
from bimvee.exportIitYarp import exportIitYarp

events_path = "/home/cappe/Desktop/uni5/Tesi/IIT/code/binaryevents/"


events = importIitYarp(filePathOrName=events_path)
exportIitYarp(events, exportFilePath=events_path, exportAsEv2 = True, exportTimestamps = False, minTimeStepPerBottle = 5e-4, viewerApp = False)