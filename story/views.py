import os
import requests
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect

import random
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

# Create your views here.
def index(request):

    def text_generator(state_dict, param_prompt, param_nsamples, param_batch_size, param_length, param_temperature, param_top_k):

        #param_prompt = "Peter was a man"
        param_quiet = False
        #param_nsamples = 1
        param_unconditional = None
        #param_batch_size = 1
        #param_length = 5
        #param_temperature = 0.95
        #param_top_k = 100

        if param_batch_size == -1:
            param_batch_size = 1
        assert param_nsamples % param_batch_size == 0

        seed = random.randint(0, 2147483647)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        enc = get_encoder()
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        model = load_weight(model, state_dict)
        model.to(device)
        model.eval()

        return param_prompt + str('... worked!')

    message = 'Error'

    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)

        try:
            param_prompt = str(request.GET.get('p', ''))
            param_nsamples = int(request.GET.get('sm', 1))
            param_batch_size = int(request.GET.get('s', 1))
            param_length = int(request.GET.get('l', random.randint(20, 125)))
            param_temperature = float(request.GET.get('v', 0.95))
            param_top_k = int(request.GET.get('i', 50000))

            try:
                #message = param_prompt, param_nsamples, param_batch_size, param_length, param_temperature, param_top_k
                message = text_generator(state_dict, param_prompt, param_nsamples, param_batch_size, param_length, param_temperature, param_top_k)
            except Exception as ex:
                message = "Please enter a prompt in the form below."

        except Exception as ex:
            message = "There was a problem. Please try again."

    return HttpResponse(message)
