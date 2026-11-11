# G13 prospective result · four-command anticipation

The spatially weighted associative SLWM passed the frozen test against ordinary dense GPT-2. The target was the net sum of four future recorded planar control commands. No future controls or outcomes enter the action head. After discovery in two earlier cohorts, the preregistration fixed the metric, two additional training seeds, the decision rule and four untouched shards. These produced **13,678 windows, 3,368 trajectories and 196 independent parent groups** after excluding source-ID or frame overlap with either prior acquisition. [Prospective protocol](prospective-action-replication.md) · [Group-level results](metrics/prospective.json)

| Training seed | Dense GPT-2 error, mm | Spatial SLWM error, mm | Paired improvement, mm (95%) |
| --- | ---: | ---: | ---: |
| 1301 | 127.06 | 118.75 | 8.31 (6.81 to 9.78) |
| 1302 | 125.44 | 124.52 | 0.92 (0.37 to 1.50) |
| 1303 | 123.51 | 119.45 | 4.06 (3.12 to 4.98) |
| 1311, new | 122.67 | 120.38 | 2.29 (1.32 to 3.31) |
| 1312, new | 124.74 | 119.98 | 4.76 (3.35 to 6.13) |

Across the five fixed seeds, the spatial processor reduced source-balanced error by **4.07 mm** (99% interval **3.31 to 4.85**). A softmax core with the same spatial bias also lost in four of five paired 95% intervals. Adding the spatial weighting to an associative core helped four seeds and hurt one. A direct raw-signal ridge was stronger than every native neural head at **111.91 mm**. Zero command scored 127.98 mm and repeating the last command scored 213.41 mm. The lower error than dense GPT-2 is a specific processor result. The raw reference fits the four-command sum directly whereas the native heads trained on the component commands. [All predictions, measured controls and held groups](metrics/prospective-actions.npz) · [Direct-readout follow-up](direct-action-readout.md)

The experiment uses real observations and teleoperator controls from Language-Table. The dataset's text is a hindsight annotation. Forecast error here does not measure the reward of executing either model's action on a physical robot. [Original dataset and methods](https://github.com/google-research/language-table)
