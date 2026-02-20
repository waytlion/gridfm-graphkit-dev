## ForecastOPF - Resume Points

### ✅ Model Code Complete
- gnn_heterogeneous_gns.py is correct
- No dimension mismatches (false alarm)
- Physics decoder receives voltages only (correct)

### Next: Task File Structure Discussion

**Questions to decide:**
1. Loss weights from plan (0.4 load, 0.3 voltage, 0.1 gen, 0.2 physics)?
2. Separate Pg/Qg losses or combined generation loss?
3. Copy opf_task.py structure or start fresh?

**After task discussion:**
- Create forecast_opf_task.py
- Create config YAML
- Test with small dataset