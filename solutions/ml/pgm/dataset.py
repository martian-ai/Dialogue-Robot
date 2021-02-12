class MyDataset(data.Dataset):
    def __init__(self, xc, xw, labels):
        self.xc = xc
        self.xw = xw
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        xc, xw, labels = self.xc[index], self.xw[index], self.labels[index]
        return xc, xw, labels

    def __len__(self):
        return len(self.xc)
