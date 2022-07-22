import logging
import logging.handlers
import multiprocessing

def setup_logging_then_call(queue, func, args):
    ''' Register a QueueHandler connected queue, then call func
    with args. Intended as a pickleable function to be given
    to Pool.starmap or Pool.apply_async.
    '''
    queue_handler = logging.handlers.QueueHandler(queue)
    queue_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    # Suppress excessive logging by plotting modules.
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    existing_handlers = list(logger.handlers)

    for handler in existing_handlers:
        logger.removeHandler(handler)

    logger.addHandler(queue_handler)

    func(*args)

    logger.removeHandler(queue_handler)

    for handler in existing_handlers:
        logger.addHandler(handler)

class PoolWithLoggerThread:
    ''' A context manager for a combination of a multiprocessing.Pool
    and a threaded handler for logging from the Pool's processes.
    '''
    
    def __init__(self, processes, logger):
        # I don't really understand multiprocessing.Queue() vs.
        # multiprocessing.Manager().Queue(), but only the latter works here.
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()

        self.queue_listener = logging.handlers.QueueListener(self.queue, *logger.handlers)

        self.pool = multiprocessing.Pool(processes=processes, maxtasksperchild=1)
        
    def apply_async(self, func, args):
        ''' Provides the same interface as Pool.starmap, but connects each
        of the Pool's processes to the logging queue before executing func.
        '''
        return self.pool.apply_async(setup_logging_then_call, (self.queue, func, args))

    def starmap(self, func, iterable):
        ''' Provides the same interface as Pool.starmap, but connects each
        of the Pool's processes to the logging queue before executing func.
        '''
        arg_tuples = ((self.queue, func, args) for args in iterable)
        self.pool.starmap(setup_logging_then_call, arg_tuples, 1)
    
    def __enter__(self):
        self.queue_listener.start()
        self.pool.__enter__()
        return self
        
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.pool.__exit__(exception_type, exception_value, exception_traceback)
        self.queue_listener.stop()

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()