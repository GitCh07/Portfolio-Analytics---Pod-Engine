# Pod Engine

A multi-strategy portfolio engine run the way a multi-manager fund runs its book: independent strategies
("pods"), an allocator that funds them by their out-of-sample record, mechanical risk limits, and a validation
gate that tests whether results are luck. It includes Value at Risk with formal backtests, an optimizer for a
portfolio you already hold (minimum variance, risk parity, Black-Litterman) and an animated single-page viewer.

**Live viewer (demo data):** https://gitch07.github.io/Portfolio-Analytics---Pod-Engine/

The viewer currently shows simulated demo data, labelled as such; it says nothing about real markets.

Full documentation, quick start and design notes: [podengine/README.md](podengine/README.md)

```bash
cd podengine
pip install -r requirements.txt
python -m podengine run --demo        # offline demo
python -m podengine run --viz         # real prices (free Yahoo Finance data, no API key), then export the viewer
python -m podengine serve             # open the viewer locally
```

Research tool, not investment advice.
