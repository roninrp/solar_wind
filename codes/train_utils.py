import torch
import torch.nn as nn
import inspect




class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs to wait for improvement before stopping.
    best_loss : float, default=0.2
        Initial best validation loss.
    min_delta : float, default=0.0
        Minimum change to qualify as an improvement.
    verbose : bool, default=False
        If True, prints progress messages.
    path : str, default="best_model.pt"
        File path to save the best model.
    """

    def __init__(self, patience=10, best_loss=0.2, min_delta=0.0, verbose=False, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = best_loss #float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss improved to {self.best_loss} → model saved to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"\rNo improvement → {self.counter}/{self.patience}", end="")
            if self.counter >= self.patience:
                self.early_stop = True



def get_optimizer_params(optimizer):
    """
    Returns a dict of hyperparameters that differ from the defaults
    for the optimizer instance passed.
    Works for Adam, SGD, RMSprop, etc.
    """
    # Get the class of the optimizer
    opt_class = optimizer.__class__
    
    # Get the signature of the constructor
    sig = inspect.signature(opt_class.__init__)
    defaults = {k: v.default for k, v in sig.parameters.items() if k != "self"}

    # Get the actual hyperparameters used
    actual = optimizer.defaults

    # Compare and keep only the ones that differ from defaults
    changed = {k: val for k, val in actual.items() if k in defaults and val != defaults[k]}
    
    return changed

def cleanup(model, optimizer):
    """
    Free GPU/XPU memory and reset seeds to ensure reproducibility.

    Parameters
    ----------
    model : torch.nn.Module
        Model to delete from memory.
    optimizer : torch.optim.Optimizer
        Optimizer to delete from memory.
    """

    try:
        del model
    except NameError:
        pass
    
    try:
        del optimizer
    except NameError:
        pass
    
    
    gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    
    # Reset Memory status
    torch.xpu.reset_peak_memory_stats()
    # torch.xpu.reset_accumulated_memory_stats() ################ --------------------------------- Used only if tracking memory profiling------------
    
    seed = 42
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)