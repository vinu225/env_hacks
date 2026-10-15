# Frontend Architecture (React + TypeScript)

This directory is designated for the user-facing web dashboard of the Hyperlocal Heat & Water Environmental Intelligence Platform.

## Technology Stack

* **Framework**: React 18+ / Vite
* **Language**: TypeScript
* **State Management**: TanStack Query (React Query) for server state caching & optimistic UI
* **Styling**: Vanilla CSS / CSS Modules (ensuring premium custom styling, vibrant tokens, and micro-animations)
* **Maps & Geo**: MapLibre GL / Leaflet for ward-level geospatial risk overlays
* **Visualizations**: Recharts / Chart.js for diurnal temperature, heat index, and rainfall curves

## Architectural Separation

The frontend is completely decoupled from backend data representations:
- Interacts exclusively with `/api/v1/*` contracts defined in `backend/app/schemas/`.
- Uses strictly typed TypeScript interfaces generated or mirrored from backend Pydantic schemas.
- Consumes composite risk scores, driver explanations, and active alerts uniformly regardless of whether the backend executed a rule baseline or an ML model.

## Planned Views

1. **Hyperlocal Hazard Map**: Real-time heat stress and inundation heatmaps for urban wards.
2. **Ward Details & Drivers**: Deep dive into wet-bulb temperature, heat index, and soil moisture indicators.
3. **Community Ground-Truth Hub**: Citizen incident reporting (waterlogging, pipe leaks, dry taps) and upvoting.
4. **Early Warning Center**: Active alert feed with actionable municipal recommendations.
5. **AI Public Advisory Panel**: Plain-language multilingual explanations powered by the backend AI service.
