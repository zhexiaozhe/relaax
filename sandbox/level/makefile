.PHONY: ps
ps:
	./master --params params.yaml --bind localhost:7000 --checkpoint-dir checkpoints

.PHONY: rlx
rlx:
	./agent --params params.yaml --bind localhost:7001 --master localhost:7000 --log-dir metrics

.PHONY: env
env:
	./env
