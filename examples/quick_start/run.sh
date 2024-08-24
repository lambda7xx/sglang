#pip install -e "python[all]"

python3 srt_example_chat.py  > srt_example_chat.log 2>&1 


# ps -aux | grep "srt_example" | grep -v grep | awk '{print $2}' | xargs kill -9