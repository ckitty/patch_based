from dataprovider import Dataprovider
from model import Model

if __name__ == '__main__':
    data = Dataprovider()
    model = Model(data,batch_size=16)
    model.train(10000)
