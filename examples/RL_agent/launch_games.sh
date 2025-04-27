#!/bin/bash

# Check if number of instances is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_instances> [start_port]"
    echo "Example: $0 3 8070"
    exit 1
fi

NUM_INSTANCES=$1
START_PORT=${2:-8070}  # Default to 8070 

echo "Launching $NUM_INSTANCES game instances starting from port $START_PORT"

# Launch each instance
for i in $(seq 0 $((NUM_INSTANCES-1))); do
    PORT=$((START_PORT + i))
    
    echo "Starting game instance $i on port $PORT"
    ./game/V1.0.5/DiceAdventure.x86_64 --args -batchmode -trainingmode -nographics -localMode reloadonend -stepTime 0 -hmtsocketurl ws://localhost -hmtsocketport $PORT &
    
    
    sleep 1
done

echo "All game instances launched. Press Ctrl+C to terminate all instances."


trap "echo 'Shutting down all instances...'; pkill -f 'DiceAdventure.x86_64'; exit" INT
wait