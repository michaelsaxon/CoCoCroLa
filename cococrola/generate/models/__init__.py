# Key the supported models short codes (used in runscripts) to their long codes (eg, HuggingFace model names)
MODEL_MAP_DIFFUSERS = {
    'SD1-1' : "CompVis/stable-diffusion-v1-1",
    'SD1-2' : "CompVis/stable-diffusion-v1-2",
    'SD1-4' : "CompVis/stable-diffusion-v1-4",
    'SD2' : "stabilityai/stable-diffusion-2",
    'SD2-1' : "stabilityai/stable-diffusion-2-1",
    'AD' : "BAAI/AltDiffusion-m9"
}

MODEL_MAP_CRAIYON = {
    'CMN' : "mini",
    'CMG' : "mega"
}

SUPPORTED_MODELS = sum([list(MODEL_MAP_DIFFUSERS.keys()), list(MODEL_MAP_CRAIYON.keys())], [])


# to_add : [ 'SD2-1', 'CV2', 'DE2' ]

def get_generator(model_code : str, device : str):
    if model_code not in SUPPORTED_MODELS:
        raise ValueError(f"Model code {model_code} not supported. Supported models are {SUPPORTED_MODELS}")
    if model_code in MODEL_MAP_DIFFUSERS:
        import diffusers
        from cococrola.generate.models.huggingface_diffusers import DiffusersImageGenerator
        if model_code == 'AD':
            pipeline_type = diffusers.AltDiffusionPipeline
        else:
            pipeline_type = diffusers.StableDiffusionPipeline
        return DiffusersImageGenerator(MODEL_MAP_DIFFUSERS[model_code], device, pipeline_type)
    elif model_code in MODEL_MAP_CRAIYON:
        from cococrola.generate.models.craiyon import CraiyonImageGenerator
        return CraiyonImageGenerator(MODEL_MAP_CRAIYON[model_code], device)