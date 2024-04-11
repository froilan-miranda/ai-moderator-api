import json
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.text.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.applications import Response
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.routing import Route

export_file_url_imdb = 'https://www.dropbox.com/scl/fi/e8oes6vu6glw1c1iusy2u/imdb_export.pkl?rlkey=a1sngtxywaqikzg5py1yrxd6o&dl=1'
export_file_url_toxic = 'https://www.dropbox.com/scl/fi/t68cwt21h0jnn7y28pi9q/toxic_comment_classifier.pkl?rlkey=7sn1f9dakmnf8myxhyqidrdv3&dl=1'
export_file_name_imdb = 'imdb_export.pkl'
export_file_name_toxic = 'toxic_comment_classifier.pkl'

path = Path(__file__).parent
# path = os.path.join(os.path.dirname(__file__), './models/')

app = Starlette()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url_toxic, path / export_file_name_toxic)
    try:
        learn = load_learner(path / export_file_name_toxic)
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
    body = await request.json()
    prediction = learn.predict(body['text'])
    print(prediction)
    #content = json.dumps(prediction[0])
    response = JSONResponse({'result': str(prediction[0])})
    return response


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5500, log_level="info")
