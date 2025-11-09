### Safety Rules Configuration

The safety rules used during analysis are **configurable at runtime** via a JSON file.

#### Location

Rule configuration lives in:

```text
src/inviol_image_analyser_assignment/services/rules/config/rules.json
```

This file defines:

* **Thresholds** for each rule (e.g. “how many people is too many?”)
* **Risk weights** used when combining rule violations into a single overall risk score

#### Format

Example `rules.json`:

```json
{
  "crowding": {
    "max_safe_people": 5,
    "high_risk_extra_people": 5,
    "medium_risk_score_per_person": 0.8,
    "high_risk_score_per_person": 1.2,
    "weight": 1.0
  },
  "person_machine_proximity": {
    "medium_risk_distance_ratio": 0.1,
    "high_risk_distance_ratio": 0.05,
    "medium_risk_score": 2.0,
    "high_risk_score": 3.0,
    "weight": 1.5
  }
}
```

#### Rules

Currently there are two rules:

1. **Crowding rule** (`crowding`)

   * Flags scenes with too many people.
   * Config fields:

     * `max_safe_people`: number of people considered safe in the scene.
     * `high_risk_extra_people`: if extra people ≥ this value, severity is **high**.
     * `medium_risk_score_per_person`: risk contribution per extra person (medium).
     * `high_risk_score_per_person`: risk contribution per extra person (high).
     * `weight`: how strongly this rule influences the overall risk score.

2. **Person–machine proximity rule** (`person_machine_proximity`)

   * Flags people standing too close to vehicles / machinery (`car`, `truck`, `bus`, `motorcycle` from COCO).
   * Distances are measured between bounding-box centres and normalised by the image diagonal.
   * Config fields:

     * `medium_risk_distance_ratio`: below this ratio -> **medium** severity.
     * `high_risk_distance_ratio`: below this ratio -> **high** severity.
     * `medium_risk_score`: risk contribution for each medium-severity violation.
     * `high_risk_score`: risk contribution for each high-severity violation.
     * `weight`: how strongly this rule influences the overall risk score.

#### How configuration is applied

* `rules.json` is loaded by `load_rules_config()` in
  `services/rules/config/rules_config.py`.
* `RuleEngine` (in `services/rule_engine.py`) reads this config **on each evaluation**:

  * It instantiates:

    * `CrowdingRule` with `CrowdingRuleConfig`
    * `PersonMachineProximityRule` with `PersonMachineProximityConfig`
  * Each rule returns one or more `RuleViolation` objects with a `risk_score`.
  * For aggregation:

    * Each violation’s `risk_score` is multiplied by the rule’s `weight`.
    * These weighted scores are summed, capped at 10, and mapped to:

      * `risk_rating` (0–10 integer)
      * `risk_level` (`low` / `medium` / `high`)

Because the config is read at evaluation time, **updating `rules.json` changes behaviour without restarting the service**.

#### Tuning examples

* Make crowding stricter:

  ```json
  "crowding": {
    "max_safe_people": 3,
    "high_risk_extra_people": 2,
    "medium_risk_score_per_person": 1.0,
    "high_risk_score_per_person": 1.5,
    "weight": 1.0
  }
  ```

* Make proximity to machinery dominate the overall risk:

  ```json
  "person_machine_proximity": {
    "medium_risk_distance_ratio": 0.12,
    "high_risk_distance_ratio": 0.06,
    "medium_risk_score": 2.0,
    "high_risk_score": 3.0,
    "weight": 2.0  // higher weight => bigger impact on overall score
  }
  ```

