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

    test = str(request.GET.get('p', ''))
    return str(test)
