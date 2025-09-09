from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
import os
import json
import time
import logging
import sys
from PIL import Image
import torch
import cv2
import traceback
import argparse

base_path = os.path.join(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))), 'third_party_models').replace('\\','/')
for i in os.listdir(base_path):
    sys.path.append(os.path.join(base_path,i).replace('\\','/'))
sys.path.append(base_path.replace('\\','/'))
sys.path.append(os.path.join(base_path, 'edited_sam2/sam2/configs/').replace('\\','/'))
sys.path.append(os.path.join(base_path, 'edited_sam2/sam2/').replace('\\','/'))
sys.path.append(os.path.join(base_path,'edited_dam4sam/dam4sam_2').replace('\\','/'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/damsam2').replace('\\','/'))

from model_scripts.run_sam import runSAM2
from model_scripts.run_florence import run_florence
from model_scripts.run_dam4sam import runDAM4SAM
from model_scripts.run_gdino import run_gdino
from transformers import AutoProcessor, AutoModelForCausalLM 

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
class MLRunner(object):
    def __init__(self, listen_dir,use_florence = False):
        self.listen_dir = listen_dir if not listen_dir.endswith('/') else listen_dir[:-1]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.job_config = None
        self.model_config_paths = './model_configs'
        self.ml_logger = logging.getLogger(__name__)
        self.model_configs = self.read_model_configs()
        self.use_florence = use_florence#
        self.prev_path = None
        self.prev_idx = None
        self.prev_limit = None
        self.is_same_path = None
        self.inference_state = None
        self.is_same_name = None
        self.name_idx = 0
        self.is_pil = True
        
        self.path_remap_base_server_path = None
        self.path_remap_replace_path = None

        # Set OCIO config path as env variable cause its easier to pass around
        os.environ['MLRUNNER_OCIOCONFIG_PATH'] = os.path.abspath(os.path.join('configs','aces1.2','config.ocio'))


        self.supported_models = {'sam':runSAM2,
                                 'florence':run_florence,
                                 'dam': runDAM4SAM,
                                 'gdino': run_gdino}
        if use_florence:
            self.ml_logger.info('Loading florence model, this will take a couple of minutes')
            self.init_florence()
            self.ml_logger.info('Loading completed!')

        self.inform_server_running()

    def init_florence(self):
        """ Load florence with ML runner as loading it takes time"""
        self.florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    def inform_server_running(self, closing = False):
        if not closing: 
            with open (os.path.join(self.listen_dir, '.server_is_running.tmp').replace('\\','/'), 'w') as f:
                f.write('server is running!')
        else:
            inform_file = os.path.join(self.listen_dir, '.server_is_running.tmp').replace('\\','/')
            if os.path.isfile(inform_file):
                os.remove(inform_file)

    def fetch_base_and_replace_path(self, config_server_path):
        """Function to detect what parts of config server path needs replacing to ensure smooth operations between OSs. 
        """
        config_list = config_server_path.split('/')
        listen_dir_list = self.listen_dir.split('/')

        # Find what parts of config path are in listen dir
        matched =[i for i in config_list if i in listen_dir_list]

        # Get whatever parts of listen dir arent in config path
        missing =  [i for i in listen_dir_list if i not in matched]

        # Get what parts need to be replaced in config path
        replace = [i for i in config_list if i not in listen_dir_list]

        # Create paths
        as_path = os.path.join(*missing)
        replace_path = os.path.join(*replace)
        
        # Ensure they are correct
        if  self.listen_dir.startswith('/') and not as_path.startswith('/'):
            as_path = '/' + as_path
        if  config_server_path.startswith('/') and not replace_path.startswith('/'):
            replace_path = '/'+replace_path

        self.path_remap_base_server_path = as_path
        self.path_remap_replace_path = replace_path

    def remap_path_to_server_os(self, path):
        """Function to fix the paths received in server. 
        This is needed because users may be sending the config from a different OS, and we need to adjust the paths to be matching on the machine that is running the server"""
        edited_path = path.replace(self.path_remap_replace_path, self.path_remap_base_server_path)
        return edited_path
        
    def read_job_config(self, filename):
        if filename.endswith('json'):
            with open(filename, 'r') as f:
                self.job_config = json.load(f)
        return self.job_config

    def read_model_configs(self):
        self.model_configs = {}
        
        for i in os.listdir(self.model_config_paths):
            if i.endswith('json'):
                with open(os.path.join(self.model_config_paths, i).replace('\\','/'), 'r') as f:
                    self.model_configs[i.replace('.json','')] = json.load(f)
        return self.model_configs

    def sam_definition(self, sam2_checkpoint_path, model_cfg_path, video_dir, frame_names, render_dir, render_name, boxes_filt, ann_frame_idx, first_frame_sequence, pred_phrases, shot_name, logger, uuid, H = None,W = None, use_gdino = True, limit_range = None, delimiter = '.'):
        sam = self.supported_models['sam'](sam2_checkpoint_path, model_cfg_path, video_dir, frame_names, render_dir, render_name, boxes_filt, ann_frame_idx, first_frame_sequence, pred_phrases, shot_name, logger, uuid, H,W, use_gdino, limit_range, self.inference_state, self.name_idx, delimiter)
        return sam
    
    def dam_definition(self, checkpoint_path, ann_frame_idx_og, video_dir, frame_names, render_dir, render_name, boxes_filt, ann_frame_idx, pred_phrases, shot_name, logger, uuid, H = None,W = None, use_gdino = True, limit_range = None):
        dam = self.supported_models['dam'](checkpoint_path, ann_frame_idx_og, video_dir, frame_names, render_dir, render_name, boxes_filt, ann_frame_idx, pred_phrases, shot_name, logger, uuid, H,W, use_gdino, limit_range)
        return dam

    def florence_definition(self, image_pil, caption, box_threshold, text_threshold = None, with_logits = True, H = None,W = None,):
        florence = self.supported_models['florence'](image_pil, self.florence_model, self.florence_processor, caption, box_threshold, text_threshold, with_logits, H, W)
        return florence
    
    def gdino_definition(self, image_pil, model_config_path, model_checkpoint_path, caption, box_threshold, text_threshold = None, with_logits = True, H = None,W = None,):
        gdino = self.supported_models['gdino'](image_pil, model_config_path, model_checkpoint_path, caption, box_threshold, text_threshold, with_logits, self.is_pil, H,W)
        return gdino

    def load_pil(self, frame_to_load):
        if '.png' in frame_to_load:
            image = Image.open(frame_to_load)
        else:
            image = cv2.imread(frame_to_load, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def get_im_width_height(self, img, is_pil = True):
        if is_pil:
            H,W = (img.height, img.width)
        else:
            H,W,C = img.shape 
        return H,W

    def get_frame_list(self, path_to_sequence, shot_name, delimiter):
        # TODO right now this works only if there is 1 sequence per folder. 
        frame_names = [
            p for p in os.listdir(path_to_sequence)
            if os.path.splitext(shot_name)[0].split(delimiter)[0] in p and p.endswith('.png') or p.endswith('.exr')   
            ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split(delimiter)[-1]))
        return frame_names
    
    def clear_sam_memory(self):
        if not self.is_same_path:
            try:
                del self.model
            except:
                self.ml_logger.error('tried to release model from memory but failed')
            self.model = None

    def keep_track(self, frame_idx, path, limit, render_name):
        self.prev_idx = frame_idx
        self.prev_path = path
        self.prev_limit = limit
        self.prev_render_name = render_name

    def run_model_based_on_cfg(self, filename):
        """Function to run a model based on a config file"""
        
        # Read config
        config = self.read_job_config(filename)
        model_to_run = config['model_to_run']
        id_class =config['id_class']
        bbox = config['crop_position']
        multisequence = config['multisequence']
        path_to_sequence = config['path_to_sequence']
        shot_name = config['shot_name']
        render_to = config['render_to']
        render_name = config['render_name']
        frame_idx = int(config['frame_idx'])
        first_frame_sequence = int(config['first_frame_sequence'])
        use_gdino = config['use_gdino']
        model_configs = self.model_configs[model_to_run]
        limit_range = config['limit_range']
        limit_first = config['limit_first']
        limit_last = config['limit_last']
        delimiter = config['delimiter']
        uuid = config['uuid']
        limit_range = (int(limit_first), int(limit_last)) if limit_range else limit_range
        self.is_pil = False if '.exr' in shot_name.lower() else True

        # Ensure paths are correct
        config_server_dir = config['listen_dir']
        config_server_dir = config_server_dir if not config_server_dir.endswith('/') else config_server_dir[:-1]
        if config_server_dir != self.listen_dir:
            # Check path differences and sets them as attributes
            self.fetch_base_and_replace_path(config_server_dir)

            # Edit all paths
            path_to_sequence = self.remap_path_to_server_os(path_to_sequence)
            render_to = self.remap_path_to_server_os(render_to)

        # Check if we are processing same footage
        self.is_same_path = path_to_sequence == self.prev_path
        self.is_same_name = frame_idx == self.prev_idx and render_name == self.prev_render_name
        self.is_same_limit = limit_range == self.prev_limit

        # If path or limit is not the same, clear inference state
        if not self.is_same_path or (not self.is_same_limit or limit_range == False):
            self.inference_state = None

        # If same name, increase index by one to avoid ovewrwriting 
        # TODO implement maybe a folder scanning to avoid rewriting with new instances of the runner
        if self.is_same_name:
            self.name_idx += 1
        else:
            self.name_idx = 0

        # Set prev frame idx and prev path, so that we can keep track of what we are doing
        self.keep_track(frame_idx, path_to_sequence, limit_range, render_name)

        H,W = None, None
        self.ml_logger.info(f'Shot {shot_name} received!')

        ## Execute model
        try:
            self.frame_names = self.get_frame_list(path_to_sequence, shot_name, delimiter)
            self.ml_logger.info(f'Frame list fetched!')
            
            if use_gdino:
                gdino_path = [i for i in sys.path if 'dino' in i.lower()][0]
                gdino_configs = os.path.join(gdino_path, self.model_configs['gdino']['config_path']).replace('\\','/')
                gdino_checkpoint = os.path.join(gdino_path, self.model_configs['gdino']['checkpoint_path']).replace('\\','/')
                try:
                    image_pil = self.load_pil(os.path.join(path_to_sequence, self.frame_names[frame_idx]).replace('\\','/'))
                except IndexError as e:
                    raise IndexError(' Could not load image for bbox detection, please check that your frame_idx is correct. Right now I have {frame_idx}')
                H,W = self.get_im_width_height(image_pil, self.is_pil)
                box_threshold = 0.35
                text_threshold = 0.25
                if self.use_florence:
                    gdino = self.florence_definition(image_pil, id_class, box_threshold, text_threshold, H,W)
                else:
                    gdino = self.gdino_definition(image_pil, gdino_configs, gdino_checkpoint, id_class, box_threshold, text_threshold, H,W)
                bbox, pred_phrases = gdino.run()
                self.ml_logger.info(f'BBox fetched!')

            else:
                # If we're not using a model like gdino/florence we need to ensure that pred phrases is set to a list == to the bbox indices
                pred_phrases = range(len(bbox))

                # Similarly, we need to read the image to fetch H,W so that we can output the mask at the right resolution
                self.ml_logger.info(f'Fetching image info..')
                try:
                    image_pil = self.load_pil(os.path.join(path_to_sequence, self.frame_names[frame_idx]).replace('\\','/'))
                    H,W = self.get_im_width_height(image_pil, self.is_pil)
                    self.ml_logger.info(f'Fetched image info')
                except IndexError as e:
                    raise IndexError(f'Failed to fetch image information. Here is the frame list info: list lenght: {len(self.frame_names)} frame index: {frame_idx}')
            
            if model_to_run == 'sam':
                print('\n\n')

                self.ml_logger.info(f'Running model {model_to_run} on shot {shot_name}')
                self.model = self.sam_definition(model_configs['checkpoint_path'],
                                                        model_configs['config_path'],
                                                        path_to_sequence, 
                                                        self.frame_names,
                                                        render_to,
                                                        render_name,
                                                        bbox, 
                                                        frame_idx, 
                                                        first_frame_sequence,
                                                        pred_phrases, 
                                                        shot_name,
                                                        self.ml_logger,
                                                        uuid,
                                                        H,
                                                        W,
                                                        use_gdino, 
                                                        limit_range, 
                                                        delimiter)
                self.inference_state = self.model.run()
                self.ml_logger.info(f'shot {shot_name} saved at {render_to}')

            elif model_to_run == 'dam':
                self.ml_logger.info(f'Running model {model_to_run} on shot {shot_name}')
                self.model = self.dam_definition(model_configs['checkpoint_path'],
                                                        first_frame_sequence,
                                                        path_to_sequence, 
                                                        self.frame_names,
                                                        render_to,
                                                        render_name,
                                                        bbox, 
                                                        frame_idx, 
                                                        pred_phrases, 
                                                        shot_name,
                                                        self.ml_logger,
                                                        uuid,
                                                        H,
                                                        W,
                                                        use_gdino,
                                                        limit_range)
                self.model.run()
                self.ml_logger.info(f'shot {shot_name} saved at {render_to}')

        except (IndexError,ValueError,TypeError, KeyError) as e:
            if isinstance(e, KeyError):
                self.ml_logger.error(f'Florence model seems to have failed. try a different frame or pass a boundibox yourself')
            self.ml_logger.error(f'An error was cought processing the config, here is the printout \n{e}')
            self.ml_logger.error(traceback.format_exc())        
        try:
            del self.model
            del bbox
            del pred_phrases
            if use_gdino:
                del gdino
        except:
            self.ml_logger.error('tried to release some memory but failed')
        
        return None


class MLRunnerHandler(FileSystemEventHandler):
    def __init__(self,runner):
        super().__init__()

        self.queue = []
        self.mlrunner = runner
        

    def add_queue(self, path):
        self.queue.append(path)
    
    def run_from_queue(self):
        for i in self.queue:
            self.mlrunner.run_model_based_on_cfg(i)
            self.queue.remove(i)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.src_path not in self.queue and event.src_path.endswith('.json'):
            self.add_queue(event.src_path)
        self.run_from_queue()

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s][%(levelname)s]: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(prog='MLRunner')
    parser.add_argument('-l', '--listen_dir', required = True, help='Directory to listen for config delivery on.')
    parser.add_argument('-f', '--use_florence', default = False, action=argparse.BooleanOptionalAction,  help='Whether to use florence or grounding dino')
    args = parser.parse_args()
    listen_path = args.listen_dir
    use_florence = args.use_florence

    assert os.path.isdir(listen_path), 'Hey the path you passed doesnt exists. Pleas insert a valid path'
    runner = MLRunner(listen_path, use_florence = use_florence)

    # Gotta use poll as the standard observer doesnt work on nfs
    is_poll = not listen_path.startswith('/mnt/')
    is_poll = not listen_path.startswith('/Volumes/') if is_poll else is_poll
    
    observer = Observer() if is_poll else PollingObserver(timeout = 1.0)
    event_handler = MLRunnerHandler(runner)

    observer.schedule(event_handler, runner.listen_dir)
    runner.ml_logger.info(f'Starting listening at directory {runner.listen_dir}')

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        runner.ml_logger.error(f'User interrupted the server, shutting down')
        runner.inform_server_running(closing=True)
        try:
            sys.exit(130)
        except SystemExit:
            os._exit(130)
    finally:
        runner.ml_logger.info(f'Shutting down server')
        runner.inform_server_running(closing=True)
        observer.stop()
        sys.exit()
        



