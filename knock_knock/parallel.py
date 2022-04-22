import logging
import logging.handlers
import multiprocessing
import threading

def setup_logging_then_call(queue, func, args):
    ''' Register a QueueHandler with queue, then call func
    with args. Intended as a pickleable function to be given
    to Pool.starmap.
    '''
    queue_handler = logging.handlers.QueueHandler(queue)
    logger = logging.getLogger()
    logger.addHandler(queue_handler)
    func(*args)
    
class PoolWithLoggerThread:
    ''' A context manager for a combination of a multiprocessing.Pool
    and a threaded handler for logging from the Pool's processes.
    '''
    
    def __init__(self, processes):
        # I don't really understand multiprocessing.Queue() vs.
        # multiprocessing.Manager().Queue(), but only the latter works here.
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()

        # Set up a thread to receive messages from the queue and log them.
        self.thread = threading.Thread(target=self.process_queue)

        self.pool = multiprocessing.Pool(processes=processes, maxtasksperchild=1)
        
    def process_queue(self):
        logger = logging.getLogger()
        
        while True:
            record = self.queue.get()
            
            if record is None:
                break
                
            logger.handle(record)
            
    def starmap(self, func, iterable):
        ''' Provides the same interface as Pool.starmap, but connects each
        of the Pool's processes to the logging queue before executing func.
        '''
        arg_tuples = ((self.queue, func, args) for args in iterable)
        self.pool.starmap(setup_logging_then_call, arg_tuples, 1)
    
    def __enter__(self):
        self.pool.__enter__()
        self.thread.start()
        return self
        
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.queue.put(None)
        self.thread.join()

        self.pool.__exit__(exception_type, exception_value, exception_traceback)