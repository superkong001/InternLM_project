from argparse import ArgumentParser

from lagent.actions import ActionExecutor, ArxivSearch, IPythonInterpreter
from lagent.agents.internlm2_agent import INTERPRETER_CN, META_CN, PLUGIN_CN, Internlm2Agent, Internlm2Protocol
from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.schema import AgentStatusCode

# from streamlit.logger import get_logger
# MODEL_DIR = "/root/ft-Oculi/merged_Oculi"
MODEL_DIR = "./telos/Oculi-InternLM2"
MODEL_NAME = "Oculi-InternLM2"


def parse_args():
    parser = ArgumentParser(description='chatbot')
    parser.add_argument(
        '--path',
        type=str,
        default=MODEL_DIR,
        help='The path to the model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Initialize the HFTransformer-based Language Model (llm)
    model = HFTransformer(
        path=args.path,
        meta_template=META,
        top_p=0.8,
        top_k=None,
        temperature=0.1,
        repetition_penalty=1.0,
        stop_words=['<|im_end|>'])
    plugin_executor = ActionExecutor(actions=[ArxivSearch()])  # noqa: F841
    interpreter_executor = ActionExecutor(actions=[IPythonInterpreter()])

    chatbot = Internlm2Agent(
        llm=model,
        plugin_executor=None,
        interpreter_executor=interpreter_executor,
        protocol=Internlm2Protocol(
            meta_prompt=META_CN,
            interpreter_prompt=INTERPRETER_CN,
            plugin_prompt=PLUGIN_CN,
            tool=dict(
                begin='{start_token}{name}\n',
                start_token='<|action_start|>',
                name_map=dict(
                    plugin='<|plugin|>', interpreter='<|interpreter|>'),
                belong='assistant',
                end='<|action_end|>\n',
            ),
        ),
    )

    def input_prompt():
        print('\ndouble enter to end input >>> ', end='', flush=True)
        sentinel = ''  # ends when this string is seen
        return '\n'.join(iter(input, sentinel))

    history = []
    while True:
        try:
            prompt = input_prompt()
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue
        if prompt == 'exit':
            exit(0)
        history.append(dict(role='user', content=prompt))
        print('\nInternLm2：', end='')
        current_length = 0
        last_status = None
        for agent_return in chatbot.stream_chat(history, max_new_tokens=512):
            status = agent_return.state
            if status not in [
                    AgentStatusCode.STREAM_ING, AgentStatusCode.CODING,
                    AgentStatusCode.PLUGIN_START
            ]:
                continue
            if status != last_status:
                current_length = 0
                print('')
            if isinstance(agent_return.response, dict):
                action = f"\n\n {agent_return.response['name']}: \n\n"
                action_input = agent_return.response['parameters']
                if agent_return.response['name'] == 'IPythonInterpreter':
                    action_input = action_input['command']
                response = action + action_input
            else:
                response = agent_return.response
            print(response[current_length:], end='', flush=True)
            current_length = len(response)
            last_status = status
        print('')
        history.extend(agent_return.inner_steps)

if __name__ == '__main__':
    if not os.path.exists(MODEL_DIR):
        from openxlab.model import download

        download(model_repo='telos/Oculi-InternLM2', output=MODEL_DIR)

        print("解压后目录结果如下：")
        print(os.listdir(MODEL_DIR))
    
    main()
