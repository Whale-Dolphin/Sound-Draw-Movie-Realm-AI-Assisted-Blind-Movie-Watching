@echo off
title SoundDrawMovieRealm-AI-AssistedBlindMovieWatching
cd Slowfast 
call activate slowfast
python .\tools\run_net.py --cfg .\demo\AVA\SLOWFAST_32x2_R101_50_50.yaml  
