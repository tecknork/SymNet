from utils import dataset, utils



if __name__ == '__main__':
    train_dataloader = dataset.get_dataloader('MIT', 'test',
                                              batchsize=1)

    print(train_dataloader.dataset.__getitem__(0))
    # test_dataloader = dataset.get_dataloader('MIT', 'test',
    #                                           batchsize=512)

