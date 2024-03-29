import json
import os
from contextlib import redirect_stdout
from typing import Any, Optional, Type

from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode

import numpy as np
import cv2
import onnxruntime as ort
from transform import resized_edge, center_crop
from streamlit.logger import get_logger

DEFAULT_DESCRIPTION = """一个眼底图像诊断的工具，
可以诊断眼底图像中的病变类型，如青光眼、是否为糖尿病视网膜病变。
输入为眼底图的图像路径，可以为本地地址，也可以为网络地址(链接)
当且仅当用户上传了图片时，才可调用本工具。
"""
logger = get_logger(__name__)


class FundusDiagnosis(BaseAction):
    def __init__(self,
                 model_path=None,
                 answer_symbol: Optional[str] = None,
                 answer_expr: Optional[str] = 'solution()',
                 answer_from_stdout: bool = False,
                 timeout: int = 20,
                 description: Optional[dict] = None,
                 parser: Type[BaseParser] = JsonParser,
                 enable: bool = True) -> None:
        super().__init__(description, parser, enable)
        self.answer_symbol = answer_symbol
        self.answer_expr = answer_expr
        self.answer_from_stdout = answer_from_stdout
        self.timeout = timeout
        
        if model_path is not None:
            assert os.path.exists(model_path), f"model_path: {model_path} not exists"
            assert model_path[-5:] == ".onnx", f"model_path: {model_path} is not a onnx model"
            self.model_path = model_path
            providers = ['CUDAExecutionProvider']

            self.model = ort.InferenceSession(model_path, providers=providers, )
        

    def __call__(self, query: str) -> ActionReturn:
        """Return the image recognition response.

        Args:
            query (str): The query include the image content path.

        Returns:recognition
            ActionReturn: The action return.
        """
        # {"image_path": "/root/GlauClsDRGrading/data/refuge/images/g0001.jpg"} 传入的是这样的字符串
        logger.info("query: " + query)
        if query.startswith("{"):
            query = query.replace("'", "\"")  # 为了解决如下错误：{'image_path':'static/lwh017-20180821-OD-1.jpg'}
            try:
                query = json.loads(query)
            except:
                t = ActionReturn(url=None, args=None, type=self.name, )
                t.result = "输入参数格式参数，输入需要为str:image_path"
                t.state = ActionStatusCode.API_ERROR
                return t
            if not (isinstance(query, dict) and ("image_path" in query or "value" in query)):
                response = "输入参数错误，请确定是否需要调用该工具"
                tool_return = ActionReturn(url=None, args=None, type=self.name)
                tool_return.result = dict(text=str(response))
                tool_return.state = ActionStatusCode.API_ERROR
                return tool_return
            if "image_path" in query:
                query = query["image_path"]
            else:
                query = query["value"]
        tool_return = ActionReturn(url=None, args=None, type=self.name)
        try:
            response = self._fundus_diagnosis(query)
            tool_return.result = dict(text=str(response))
            tool_return.state = ActionStatusCode.SUCCESS
        except Exception as e:
            tool_return.result = dict(text=str(e))
            tool_return.state = ActionStatusCode.API_ERROR
        return tool_return

    def _fundus_diagnosis(self, query: str) -> str:
        logger.info("Enter Image Recognition entry\n\n\n\ns")
        # data = json.loads(query)

        image_path = query
        logger.info("查询是: " + query)
        if not os.path.exists(image_path):
            return "由于图片路径无效，无法进行有效诊断"
        img = cv2.imread(image_path)

        img = resized_edge(img, 448, edge='long')
        img = center_crop(img, 448)
        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
        img = (img - mean) / std
        img = img[..., ::-1]  # bgr to rgb
        img = img.transpose((2, 0, 1))
        img = img.astype('float32')
        img = img[np.newaxis, ...]

        output = self.model.run(None, {'input': img})

        glaucoma = output[0][0].argmax()
        dr = output[1][0].argmax()
        res = ""
        if glaucoma == 0 and dr == 0:
            res = '这张图表明您的眼睛状况良好，无青光眼和糖尿病视网膜病变'
        elif glaucoma == 1 and dr == 0:
            res = '这张图表明您是一个青光眼患者, 但无糖尿病视网膜病变'
        elif glaucoma == 0 and dr >= 1:
            res = '这张图表明您不是一个青光眼患者，但是糖尿病视网膜病变患者, 且病变程度为' + str(dr)
        elif glaucoma == 1 and dr >= 1:
            res = '这张图表明您是一个青光眼患者, 并且患有糖尿病视网膜病变，且病变程度为' + str(dr)
        return res
