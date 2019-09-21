from pathlib import Path
from fastai.vision import *
import types
import cv2 as cv

def load_model():
    pass

def evaluate_input(img):
    pass
    # data = cv.imencode('.png', img)[1].tobytes()
    # return data

##################################################################

from pathlib import Path

class NormalizationProcessor(PreProcessor):
    "`PreProcessor` that computes mean and std from `ds.items` and normalizes them."
    def __init__(self, ds:ItemList):        
        self.compute_stats(ds)
        
        self.state_attrs = ['mean', 'std']
        
    def compute_stats(self, ds:ItemList):
        items = ds.items[~np.isnan(ds.items)]
        self.mean = items.mean()
        self.std = items.std()

    def process_one(self,item):
        if isinstance(item, EmptyLabel): return item
        return (item - self.mean) / self.std
    
    def unprocess_one(self, item):
        if isinstance(item, EmptyLabel): return item
        return item * self.std + self.mean
        
    def process(self, ds):
        if self.mean is None: 
            self.compute_stats(ds)
        ds.mean = self.mean
        ds.std = self.std
        super().process(ds)

    def __getstate__(self): 
        return {n:getattr(self,n) for n in self.state_attrs}

    def __setstate__(self, state:dict):
        self.state_attrs = state.keys()
        for n in state.keys():
            setattr(self, n, state[n])
                

class NormalizedFloatList(FloatList):
    _processor = NormalizationProcessor

def multitask_loss(inputs_concat, *targets, **kwargs):
  mt_lengths, mt_types = data.mt_lengths, data.mt_types
  start = 0
  
  loss_size = targets[0].shape[0] if kwargs.get('reduction') == 'none' else 1
  losses = torch.zeros([loss_size]).cuda()
  
  for i, length in enumerate(data.mt_lengths):
      
      input = inputs_concat[:,start: start + length]
      target = targets[i]
      
      input, target = _clean_nan_values(input, target, data.mt_types[i], data.mt_classes[i])
      
      if data.mt_types[i] == CategoryList:
          losses += CrossEntropyFlat(**kwargs)(input, target).cuda()
      elif issubclass(data.mt_types[i], FloatList):
          losses += MSELossFlat(**kwargs)(input, target).cuda()
      start += length
  
  if kwargs.get('reduction') == 'none':
      return losses
  return losses.sum()
class MultitaskAverageMetric(AverageMetric):
    def __init__(self, func, name=None):
        super().__init__(func)
        self.name = name # subclass uses this attribute in the __repr__ method.

def _mt_parametrable_metric(inputs, *targets, func, start=0, length=1, i=0):
    input = inputs[:,start: start + length]
    target = targets[i]

    _remove_nan_values(input, target, data.mt_types[i], data.mt_classes[i])
    
    if func.__name__ == 'root_mean_squared_error':
        processor = listify(learn.data.y.processor)
        input = processor[0].procs[i][0].unprocess_one(input) 
        target = processor[0].procs[i][0].unprocess_one(target.float()) 
    return func(input, target)

def _format_metric_name(field_name, metric_func):
    return f"{field_name} {metric_func.__name__.replace('root_mean_squared_error', 'RMSE')}"

def mt_metrics_generator(multitask_project, mt_lengths):
    metrics = []
    start = 0
    for i, ((name, task), length) in enumerate(zip(multitask_project.items(), mt_lengths)):
        metric_func = task.get('metric')
        if metric_func:
            partial_metric = partial(_mt_parametrable_metric, start=start, length=length, i=i, func=metric_func)
            metrics.append(MultitaskAverageMetric(partial_metric, _format_metric_name(name,metric_func)))
        start += length
    return metrics

# Monkey patch FloatItem with a better default string formatting.
def float_str(self):
    return "{:.4g}".format(self.obj)
FloatItem.__str__ = float_str


def _processor_unprocess_one(self, item:Any):
    res = []
    for procs, i in zip(self.procs, item):
        for p in procs: 
            if hasattr(p, 'unprocess_one'):
                i = p.unprocess_one(i)
        res.append(i)
    return res


class MultitaskItem(MixedItem):    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_names = mt_names
    
    def __repr__(self):
        return '|'.join([f'{self.mt_names[i]}:{item}' for i, item in enumerate(self.obj)])

class MultitaskItemList(MixedItemList):
    
    def __init__(self, *args, mt_names=None, **kwargs):
        super().__init__(*args,**kwargs)
        self.mt_classes = [getattr(il, 'classes', None) for il in self.item_lists]
        self.mt_types = [type(il) for il in self.item_lists]
        self.mt_lengths = [len(i) if i else 1 for i in self.mt_classes]
        self.mt_names = mt_names
        
    def get(self, i):
        return MultitaskItem([il.get(i) for il in self.item_lists], mt_names=self.mt_names)
    
    def reconstruct(self, t_list):
        items = []
        t_list = self.unprocess_one(t_list)
        for i,t in enumerate(t_list):
            if self.mt_types[i] == CategoryList:
                items.append(Category(t, self.mt_classes[i][t]))
            elif issubclass(self.mt_types[i], FloatList):
                items.append(FloatItem(t))
        return MultitaskItem(items, mt_names=self.mt_names)
    
    def analyze_pred(self, pred, thresh:float=0.5):         
        predictions = []
        start = 0
        for length, mt_type in zip(self.mt_lengths, self.mt_types):
            if mt_type == CategoryList:
                predictions.append(pred[start: start + length].argmax())
            elif issubclass(mt_type, FloatList):
                predictions.append(pred[start: start + length][0])
            start += length
        return predictions

    def unprocess_one(self, item, processor=None):
        if processor is not None: self.processor = processor
        self.processor = listify(self.processor)
        for p in self.processor: 
            item = _processor_unprocess_one(p, item)
        return item

class MultitaskLabelList(LabelList):
    def get_state(self, **kwargs):
        kwargs.update({
            'mt_classes': self.mt_classes,
            'mt_types': self.mt_types,
            'mt_lengths': self.mt_lengths,
            'mt_names': self.mt_names
        })
        return super().get_state(**kwargs)

    @classmethod
    def load_state(cls, path:PathOrStr, state:dict) -> 'LabelList':
        res = super().load_state(path, state)
        res.mt_classes = state['mt_classes']
        res.mt_types = state['mt_types']
        res.mt_lengths = state['mt_lengths']
        res.mt_names = state['mt_names']
        return res
    
class MultitaskLabelLists(LabelLists):
    @classmethod
    def load_state(cls, path:PathOrStr, state:dict):
        path = Path(path)
        train_ds = MultitaskLabelList.load_state(path, state)
        valid_ds = MultitaskLabelList.load_state(path, state)
        return MultitaskLabelLists(path, train=train_ds, valid=valid_ds)


def np2Image(frame):
  return vision.image.Image(pil2tensor(frame, np.float32).div_(255))

def mt_load_learner(path:PathOrStr, file:PathLikeOrBinaryStream='export.pkl', test:ItemList=None, **db_kwargs):
  
    source = Path(path)/file if is_pathlike(file) else file
    state = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
    model = state.pop('model')
    src = MultitaskLabelLists.load_state(path, state.pop('data'))

    if test is not None: src.add_test(test)
    data = src.databunch(**db_kwargs)
    data.single_ds.y.mt_classes = src.mt_classes
    data.single_ds.y.mt_lengths = src.mt_lengths
    data.single_ds.y.mt_types = src.mt_types
    data.single_ds.y.mt_names = src.mt_names
    
    cb_state = state.pop('cb_state')
    clas_func = state.pop('cls')
    res = clas_func(data, model, **state)
    res.callback_fns = state['callback_fns'] #to avoid duplicates
    res.callbacks = [load_callback(c,s, res) for c,s in cb_state.items()]
    return res