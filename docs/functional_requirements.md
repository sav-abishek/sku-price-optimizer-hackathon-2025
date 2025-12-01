[Refer this docx file for Complete functional requirement](https://anheuserbuschinbev-my.sharepoint.com/:w:/r/personal/pearlnova_antony_ab-inbev_com/Documents/My%20Documents/instructions.docx?d=w9a258a87d64b4911bd76874b7c6057c0&csf=1&web=1&e=DhIJqF)

## Common:

- Tabular rendering for the optimization results.
- Allow filters for columns
- Export functionality: CSV or Excel of scenario inputs + predicted power. 
- Responsive UI: usable on both desktop and laptop screens. 

### __LTE__: 
- Filters to select country, brand, and time period (monthly, quarterly, etc.) - even if monthly, it should use the R1 model aggregation to roll up to quarterly and show quarterly results 
- Dropdown or search functionality for each. 
- List all marketing KPIs (X variables) used in the model. Users can adjust each KPI via: 
- Numeric input box (e.g., type exact value) 
- Slider (for incremental changes, e.g., ±20% of baseline) 
- Display the predicted power value based on current KPI settings.
- Ability to save multiple scenarios for the same country-brand-period combination. 
- Compare predicted power across scenarios (e.g., table or chart). 
- Highlight delta from baseline (current predicted power vs. scenario). 
- Reset button to restore baseline KPI values. 

### __RGM__:
- Take user input on PINC and Price Bound > Optimize for MACO > recommend optimal price/SKU adhering to constraints 
- Tabular visualization of the overall Portfolio Impact and Financial summary. This table should clearly show – for the given user input; the old price, new price, and the projected impact (%change) on key performance indicators (KPIs), including: 
 - Volume 
 - Net Revenue (NR) 
 - NR per Hectoliter (NR/HL) 
 - MACO
 - MACO per Hectoliter (MACO/HL) 

- Price Architecture View - A visualization of the new and old prices across SKUs based on the recommendations from the optimizer for the user defined PINC, that can be viewed by product attributes like brand, pack, size.  