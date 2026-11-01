-- CreditRisk AI / RAG Phase 1
-- Safe additive indexes for the existing credit_risk_history table.
-- This script does not alter or delete existing records.

CREATE INDEX IF NOT EXISTS idx_credit_risk_history_created_at
    ON public.credit_risk_history (created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_credit_risk_history_grade
    ON public.credit_risk_history ((upper(coalesce(payload->>'grade', ''))));

CREATE INDEX IF NOT EXISTS idx_credit_risk_history_mode
    ON public.credit_risk_history ((coalesce(payload->>'mode', '')));

-- Optional expression index for numeric PD. The CASE guards malformed/missing values.
CREATE INDEX IF NOT EXISTS idx_credit_risk_history_pd
    ON public.credit_risk_history (
        (CASE
            WHEN coalesce(payload->>'pd_value', '') ~ '^[0-9]+([.][0-9]+)?$'
                THEN (payload->>'pd_value')::numeric
            WHEN coalesce(payload->>'pd_display', '') ~ '^[0-9]+([.][0-9]+)?%$'
                THEN replace(payload->>'pd_display', '%', '')::numeric / 100
            ELSE NULL
         END)
    );
