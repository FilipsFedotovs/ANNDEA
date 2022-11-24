
import pickle
import UtilityFunctions as UF


def write(file, data):
    pickle_writer=open(file,"wb")
    pickle.dump(data, pickle_writer)
    pickle_writer.close()
    print("UF.PickleOperations Message: Data has been written successfully into "+file)

def load(file):
    print(UF.TimeStamp(),'Loading the data file ',bcolors.OKBLUE+TrainSampleInputMeta+bcolors.ENDC)
    pickle_reader=open(file,'rb')
    data=pickle.load(pickle_reader)
    pickle_reader.close()
    print("UF.PickleOperations Message: Data has been loaded successfully from "+file)
    return data

