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
                message = param_prompt, param_nsamples, param_batch_size, param_length, param_temperature, param_top_k
                #message = text_generator(state_dict, param_prompt, param_nsamples, param_batch_size, param_length, param_temperature, param_top_k)
            except Exception as ex:
                message = "Please enter a prompt in the form below."

        except Exception as ex:
            message = "There was a problem. Please try again."

    return HttpResponse(message)
