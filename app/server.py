from starlette.applications import Response
from starlette.responses import JSONResponse
from starlette.routing import Route
import json
import aiohttp

import asyncio
import uvicorn
from fastai import *
from fastai.text.all import *
# from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/scl/fi/e8oes6vu6glw1c1iusy2u/imdb_export.pkl?rlkey=a1sngtxywaqikzg5py1yrxd6o&dl=1'
export_file_name = 'imdb_export.pkl'

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path / export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    # img_data = await request.form()
    # img_bytes = await (img_data['file'].read())
    # img = open_image(BytesIO(img_bytes))
    # prediction = learn.predict(img)[0]
    # return JSONResponse({'result': str(prediction)})
    body = await request.json()
    prediction = learn.predict(body['text'])
    print(prediction)
    content = json.dumps(prediction[0])
    # response = Response(content,status_code=200, headers=None, media_type='application/json')
    response = JSONResponse({'result': str(prediction)})
    return response


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5500, log_level="info")
