import torch
import torch.nn.functional as F
import torch.utils.data as torchdata
import ignite.engine as engine
from ignite.engine import Events
import ignite.metrics as ignite_metrics
from ignite.handlers import  EarlyStopping
import tqdm
import time
import sys
import numpy as np
from ignite.utils import convert_tensor

class WindowedDataset(torchdata.Dataset):
    def __init__(self, xdata,ydata,window):
        self.xdata = xdata
        self.ydata = ydata
        self.window = window

    def __getitem__(self, index):
        x = self.xdata[index:index+self.window]
        y = self.ydata[self.window+index-1]
        return x,y

    def __len__(self):
        return len(self.xdata) - self.window

class WindowedDataset_with_weights(torchdata.Dataset):
    def __init__(self, xdata,ydata,weights,window):
        self.xdata = xdata
        self.ydata = ydata
        self.weights = weights
        self.window = window

    def __getitem__(self, index):
        x = self.xdata[index:index+self.window]
        y = self.ydata[self.window+index-1]
        z = self.weights[self.window+index-1]
        return x,y,z

    def __len__(self):
        return len(self.xdata) - self.window
    
def count_parameters(model):
    '''
    Count the number of model parameters.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_batched_windowed_data_from_timeseries(x,y,idx_train,idx_val,idx_test,
                                                   window_length,batch_size,sample_weights=None):
   
    test_dataloader = torchdata.DataLoader(WindowedDataset(x[idx_test],y[idx_test],window_length),batch_size=batch_size,shuffle=False) #don't need loss weights for test set
    
    if sample_weights is not None: #if weighting loss according to sample weights, we need dataloaders to include sample weights for train and val splits
        train_dataloader = torchdata.DataLoader(WindowedDataset_with_weights(x[idx_train],y[idx_train],sample_weights[idx_train],window_length),batch_size=batch_size,shuffle=False)
        val_dataloader = torchdata.DataLoader(WindowedDataset_with_weights(x[idx_val],y[idx_val],sample_weights[idx_val],window_length),batch_size=batch_size,shuffle=False)
        
        example_batch_x, example_batch_y, example_batch_z = next(iter(train_dataloader))
        return train_dataloader, val_dataloader, test_dataloader, example_batch_x, example_batch_y, example_batch_z
    
    else:
        train_dataloader = torchdata.DataLoader(WindowedDataset(x[idx_train],y[idx_train],window_length),batch_size=batch_size,shuffle=False)
        val_dataloader = torchdata.DataLoader(WindowedDataset(x[idx_val],y[idx_val],window_length),batch_size=batch_size,shuffle=False)
    
        example_batch_x, example_batch_y = next(iter(train_dataloader))
        return train_dataloader, val_dataloader, test_dataloader, example_batch_x, example_batch_y

    
def generate_torch_batched_data(x_train, y_train, x_test, y_test, train_batch_size, test_batch_size,
                                x_valid=None, y_valid=None, valid_batch_size=None,
                                shuffle=False, num_workers=0): #original version Chiheb & student
    '''
    Generate torch dataloaders. (There is no validation dataset by default.)
    Arg:
        - x(y)_train(valid, test): shape can be (batch, seq, feature) or (batch, feature).
        - shuffle: set to True to have the data reshuffled at every epoch (default: False).
        - num_workers: how many subprocesses to use for data loading (default: 0).
    '''
    # 'TensorDataset' 将输入数据和目标数据组合成一个数据集对象
    # 'Dataloader' 用于创建数据加载器， 在迭代训练时按批次提供数据
    train_dataset = torchdata.TensorDataset(torch.tensor(x_train, dtype=torch.float),
                                            torch.tensor(y_train, dtype=torch.float))
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=train_batch_size,
                                            shuffle=shuffle, num_workers=num_workers)
    test_dataset = torchdata.TensorDataset(torch.tensor(x_test, dtype=torch.float),
                                           torch.tensor(y_test, dtype=torch.float))
    test_dataloader = torchdata.DataLoader(test_dataset, batch_size=test_batch_size,
                                           shuffle=shuffle, num_workers=num_workers)
    # 用于初始化模型
    example_batch_x, example_batch_y = next(iter(train_dataloader))
    if x_valid is not None and y_valid is not None:
        valid_dataset = torchdata.TensorDataset(torch.tensor(x_valid, dtype=torch.float),
                                                torch.tensor(y_valid, dtype=torch.float))
        valid_dataloader = torchdata.DataLoader(valid_dataset, batch_size=valid_batch_size,
                                                shuffle=shuffle, num_workers=num_workers)
        return train_dataloader, valid_dataloader, test_dataloader, example_batch_x, example_batch_y
    else:
        return train_dataloader, test_dataloader, example_batch_x, example_batch_y
    
def create_model_supervised_trainer(lr, optimizer_fn, loss_fn, weight_loss, max_epochs, patience, train_dataloader, eval_dataloader, example_batch_x):
    '''
    To save and load the trained model:
        # save
        torch.save(model.state_dict(), 'my_model.pth')
        # load
        model.load_state_dict(torch.load('my_model.pth'))
        # switch the model to evaluation mode
        model.eval()
        # prediction
        prediction = model(input_data)
    Args:
        - loss_fn: 可自定的损失函数，不用于训练。
    '''
    def train_model(model, model_name, history, device=None):
        model(example_batch_x)# 初始化模型
        optimizer = optimizer_fn(model.parameters(), lr=lr)
        if weight_loss==True:
            history[model_name] = {'train_loss': [], 'eval_loss': []} #cannot evaluate mse if passing weights through y as a tuple of (y_true,weights)
        else:
            history[model_name] = {'train_loss': [], 'train_mse': [], 'eval_loss': [], 'eval_mse': []}
        if device not in ('cuda', 'cpu'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 定义训练器和评估器
        if weight_loss:
            def prepare_batch_with_weights(batch, device, non_blocking): #adapt prepare_batch to handle 3 tensors in batch
                x,y,z = batch  # get x from batch
                
                return (
                    convert_tensor(x, device, non_blocking),
                    (convert_tensor(y, device, non_blocking), #pass combination of observations and weights as a tuple to the loss function (see https://github.com/pytorch/ignite/issues/361))
                    convert_tensor(z, device, non_blocking)), #3 outputs cannot be handled by ignite evaluator
                )
            
            trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, device=device, prepare_batch=prepare_batch_with_weights)
            evaluator = engine.create_supervised_evaluator(model, device=device, prepare_batch=prepare_batch_with_weights, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                          })
            
            val_evaluator = engine.create_supervised_evaluator(model, device=device, prepare_batch=prepare_batch_with_weights, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                          }) #need a separate evaluator for val split if using early stopping on val loss
        else:
            trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, device=device)
            evaluator = engine.create_supervised_evaluator(model, device=device, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                          'mse': ignite_metrics.MeanSquaredError()})
            
            val_evaluator = engine.create_supervised_evaluator(model, device=device, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                          'mse': ignite_metrics.MeanSquaredError()}) #need a separate evaluator for val split if using early stopping on val loss
            
        handler = EarlyStopping(patience=patience, score_function=lambda engine: engine.state.metrics['loss']*-1, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)
        ###
        desc = "Epoch: {:4}{:12}"
        num_batches = len(train_dataloader)

        @trainer.on(engine.Events.STARTED)
        def logging_results(trainer):
            # training process
            evaluator.run(train_dataloader)
            if weight_loss == False:
                train_mse = evaluator.state.metrics['mse']
            train_loss = evaluator.state.metrics['loss']
            # evaluating process
            val_evaluator.run(eval_dataloader)
            if weight_loss == False:
                eval_mse = val_evaluator.state.metrics['mse']
            eval_loss = val_evaluator.state.metrics['loss']
        
            
            # logging
            if weight_loss == False:
                tqdm.tqdm.write("train mse: {:5.4f} --- train loss: {:5.4f} --- eval mse: {:5.4f} --- eval loss: {:5.4f}"
                            .format(train_mse, train_loss, eval_mse, eval_loss), file=sys.stdout)
            else:
                tqdm.tqdm.write("train loss: {:5.4f} --- eval loss: {:5.4f}"
                            .format(train_loss, eval_loss), file=sys.stdout)
            model_history = history[model_name]
            if weight_loss == False:
                model_history['train_mse'].append(train_mse); model_history['train_loss'].append(train_loss)
                model_history['eval_mse'].append(eval_mse); model_history['eval_loss'].append(eval_loss)
            else:
                model_history['train_loss'].append(train_loss)
                model_history['eval_loss'].append(eval_loss)
        @trainer.on(engine.Events.EPOCH_STARTED)
        def creat_pbar(trainer):
            trainer.state.pbar = tqdm.tqdm(initial=0, total=num_batches, desc=desc.format(trainer.state.epoch, ''),
                                           file=sys.stdout)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def logging_results_(trainer):
            trainer.state.pbar.n = num_batches
            trainer.state.pbar.last_print_n = num_batches
            trainer.state.pbar.refresh()
            trainer.state.pbar.close()
            logging_results(trainer)

        start = time.time()
        trainer.run(train_dataloader, max_epochs=max_epochs)
        end = time.time()
        tqdm.tqdm.write("Training took {:.2f} seconds.".format(end - start), file=sys.stdout)
    return train_model


def create_model_supervised_trainer_with_weights(lr, optimizer_fn, loss_fn, max_epochs, patience, train_dataloader, eval_dataloader, example_batch_x):
    '''
    To save and load the trained model:
        # save
        torch.save(model.state_dict(), 'my_model.pth')
        # load
        model.load_state_dict(torch.load('my_model.pth'))
        # switch the model to evaluation mode
        model.eval()
        # prediction
        prediction = model(input_data)
    Args:
        - loss_fn: 可自定的损失函数，不用于训练。
    '''
  
    #def loss_function(y_pre,y):
    #    '''
    #    默认使用MSE作为训练的损失函数。
    #    '''
    #    return F.mse_loss(y_pre,y)

    def train_model(model, model_name, history, device=None):
        model(example_batch_x)# 初始化模型
        optimizer = optimizer_fn(model.parameters(), lr=lr)
        history[model_name] = {'train_loss': [], 'train_mse': [], 'eval_loss': [], 'eval_mse': []}
        if device not in ('cuda', 'cpu'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 定义训练器和评估器
        
        def prepare_batch_fn(batch, device, non_blocking):
            x,y,z = batch  # get x from batch
            
            return (
                convert_tensor(x, device, non_blocking),
                (convert_tensor(y, device, non_blocking),
                convert_tensor(z, device, non_blocking)),
            )
        
  
        '''
        def _prepare_batch(
            batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
        ) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
            """Prepare batch for training or evaluation: pass to a device with options."""
            x, y = batch
            return (
        convert_tensor(x, device=device, non_blocking=non_blocking),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )
        '''
        def output_transform_fn(x,y,y_pred):
            return (y_pred,y)


        trainer = engine.create_supervised_trainer(model, optimizer, loss_fn, device=device,prepare_batch=prepare_batch_fn)
        evaluator = engine.create_supervised_evaluator(model, device=device, prepare_batch=prepare_batch_fn, 
                                                                                   metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                      },output_transform=output_transform_fn)
        
        ###TH:
        val_evaluator = engine.create_supervised_evaluator(model, device=device, prepare_batch=prepare_batch_fn, metrics={'loss': ignite_metrics.Loss(loss_fn),
                                                                                      })
        
        handler = EarlyStopping(patience=patience, score_function=lambda engine: engine.state.metrics['loss']*-1, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)
        ###
        desc = "Epoch: {:4}{:12}"
        num_batches = len(train_dataloader)

        @trainer.on(engine.Events.STARTED)
        def logging_results(trainer):
            # training process
            evaluator.run(train_dataloader)
            #train_mse = evaluator.state.metrics['mse']
            train_loss = evaluator.state.metrics['loss']
            # evaluating process
            val_evaluator.run(eval_dataloader)
            #eval_mse = val_evaluator.state.metrics['mse']
            eval_loss = val_evaluator.state.metrics['loss']
            '''
            evaluator.run(eval_dataloader)
            eval_mse = evaluator.state.metrics['mse']
            eval_loss = evaluator.state.metrics['loss']
            '''
            # logging
            tqdm.tqdm.write("train loss: {:5.4f} --- eval loss: {:5.4f}"
                            .format(train_loss, eval_loss), file=sys.stdout)
            model_history = history[model_name]
          
        @trainer.on(engine.Events.EPOCH_STARTED)
        def creat_pbar(trainer):
            trainer.state.pbar = tqdm.tqdm(initial=0, total=num_batches, desc=desc.format(trainer.state.epoch, ''),
                                           file=sys.stdout)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def logging_results_(trainer):
            trainer.state.pbar.n = num_batches
            trainer.state.pbar.last_print_n = num_batches
            trainer.state.pbar.refresh()
            trainer.state.pbar.close()
            logging_results(trainer)

        start = time.time()
        trainer.run(train_dataloader, max_epochs=max_epochs)
        end = time.time()
        tqdm.tqdm.write("Training took {:.2f} seconds.".format(end - start), file=sys.stdout)
    return train_model
