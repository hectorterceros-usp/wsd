# -*- coding: utf-8 -*-
# Script para isolar as medidas de similaridade

def lesk(x, y):
    return len(
        set(x.definition().split()) &
        set(y.definition().split()))
