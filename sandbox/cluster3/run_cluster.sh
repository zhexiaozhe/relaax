N=$1
PIDS=()

echo master
source activate server&&exec python master.py --bind localhost:7000 --checkpoint-dir checkpoints/boxing_a3c_8_agents &>out/master &
PIDS+=($!)
sleep 1

echo agent
source activate server&&exec python agent.py --bind localhost:7001 --master localhost:7000 --log-dir logs/boxing_a3c_8_agents &>out/agent &
PIDS+=($!)
sleep 1

for i in `seq 0 $((N - 1))`;
do
    echo client $i
    source activate client&&exec python ../../clients/rl_client_ale.py --agent localhost:7001 --seed $i &>out/client_$i &
    PIDS+=($!)
    sleep 1
done

read -p "Press [Enter] key to stop cluster..."

for i in `seq $((${#PIDS[@]} - 1)) -1 0`;
do
    echo stop $((PIDS[i]))
    kill -SIGINT $((PIDS[i]))
    sleep 1
done

sleep 2

ps ax | grep python
