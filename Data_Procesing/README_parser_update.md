# Parser update notes

Run from the project root:

```powershell
python Data_Procesing\Raw_Data_Parser.py
```

or from inside `Data_Procesing`:

```powershell
python Raw_Data_Parser.py
```

The parser now prefers configuration sources in this order:

1. The API log beside the selected `.bin`, for example `ADC_Data_LogFile.txt`.
2. `mmWave_Configuration/Profile.csv` for human-readable profile values.
3. An explicit XML if passed with `--config`, or an XML inside `mmWave_Configuration`.
4. Root-level XML only if you pass `--allow-root-xml-fallback`.

This prevents stale root-level XML files from overriding the actual capture settings.

Useful commands:

```powershell
python Data_Procesing\Raw_Data_Parser.py --dry-run
python Data_Procesing\inspect_parsed.py
python Data_Procesing\plot_rd.py
```

If automatic file selection is ambiguous, pass explicit paths:

```powershell
python Data_Procesing\Raw_Data_Parser.py ^
  --bin ADC_Recorded_Data\ADC_Data.bin ^
  --log ADC_Recorded_Data\ADC_Data_LogFile.txt ^
  --profile-csv mmWave_Configuration\Profile.csv
```
