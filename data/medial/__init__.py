#!/usr/bin/env false
# -*- coding: utf-8 -*-

from .rectangle import MedialRectangle

def data_factory(name: str, **kwargs):
    if name.casefold() == "rectangle":
        return MedialRectangle(**kwargs)
    else:
        raise ValueError(f"Unknown medial data name: {name}")
