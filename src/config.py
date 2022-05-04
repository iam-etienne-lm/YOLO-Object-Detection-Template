"""
CONFIG
Defines global variables
os can be used to define global root path (can be a source of issues for tests)
"""
import os
# import logging
import re

# App settings
name = "spaghetti app"

host = "0.0.0.0"

port = int(os.environ.get("PORT", 5000))

debug = True

fontawesome = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'


#
# END CONFIG
#


contacts = "#"

code = "#"

tutorial = "#"