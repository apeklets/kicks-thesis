import os

from datetime import datetime
from .generation_tests import *

from utils.utils import mkdir_in_path, load_model_checkp, saveAudioBatch
from .generation_tests import StyleGEvaluationManager
from data.preprocessing import AudioPreprocessor


def generate(parser):
    args = parser.parse_args()

    model, config, model_name = load_model_checkp(**vars(args))
    latentDim = model.config.categoryVectorDim_G

    # We load a dummy data loader for post-processing
    postprocess = AudioPreprocessor(**config['transformConfig']).get_postprocessor()

    # Create output evaluation dir
    output_dir = mkdir_in_path(args.dir, f"generation_tests")
    output_dir = mkdir_in_path(output_dir, model_name)
    output_dir = mkdir_in_path(output_dir, "random")
    output_dir = mkdir_in_path(output_dir, datetime.now().strftime('%Y-%m-%d%H%M'))
    
    
    # Create evaluation manager
    eval_manager = StyleGEvaluationManager(model, n_gen=250)

    gen_batch = eval_manager.test_random_generation()
    audio_out = map(postprocess, gen_batch)

    saveAudioBatch(audio_out,
                   path=output_dir,
                   basename='random', 
                   sr=config["transformConfig"]["sample_rate"])
    print("FINISHED!\n")