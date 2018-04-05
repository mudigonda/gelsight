
Requires Python 3

## Run experiments:
```
python reacher_pd.py
```

## Compile Docker
```
docker build -t rcalandra/pytorch .
```

## Blender

Start Blender server
```
blender /home/rcalandra/Dropbox/Research/tactile-servo/gelsight.blend --python blender_server.py

```

Send commands to the server
```
python blender_client.py blender.py
```