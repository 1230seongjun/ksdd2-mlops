--
-- PostgreSQL database dump
--

\restrict iSpkLFVLvrY2QQSn8o2C6fiwhFTpaCNipwt9EVGNsTnPduMEIJyD4bjXexdLl46

-- Dumped from database version 16.15 (Debian 16.15-1.pgdg13+2)
-- Dumped by pg_dump version 16.15 (Debian 16.15-1.pgdg13+2)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: classifier_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.classifier_results (
    request_id uuid NOT NULL,
    model_name text NOT NULL,
    model_version integer NOT NULL,
    defect_type text,
    defect_subtype text,
    painted_pixels integer,
    latency_ms integer,
    CONSTRAINT classifier_results_defect_type_check CHECK ((defect_type = ANY (ARRAY['brown'::text, 'black'::text, 'blue'::text, 'surface'::text])))
);


--
-- Name: detector_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.detector_results (
    request_id uuid NOT NULL,
    model_name text NOT NULL,
    model_version integer NOT NULL,
    anomaly_score double precision NOT NULL,
    threshold double precision NOT NULL,
    is_anomaly boolean NOT NULL,
    latency_ms integer
);


--
-- Name: inference_requests; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.inference_requests (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    filename text NOT NULL,
    image_sha256 character(64) NOT NULL,
    model_choice text NOT NULL,
    prediction text NOT NULL,
    total_latency_ms integer,
    batch_id uuid,
    image_uri text,
    reviewed_label text,
    reviewed_at timestamp with time zone,
    CONSTRAINT inference_requests_model_choice_check CHECK ((model_choice = ANY (ARRAY['current'::text, 'latest'::text, 'best_recall'::text]))),
    CONSTRAINT inference_requests_prediction_check CHECK ((prediction = ANY (ARRAY['normal'::text, 'defect'::text])))
);


--
-- Name: classifier_results classifier_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classifier_results
    ADD CONSTRAINT classifier_results_pkey PRIMARY KEY (request_id);


--
-- Name: detector_results detector_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detector_results
    ADD CONSTRAINT detector_results_pkey PRIMARY KEY (request_id);


--
-- Name: inference_requests inference_requests_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.inference_requests
    ADD CONSTRAINT inference_requests_pkey PRIMARY KEY (id);


--
-- Name: idx_classifier_defect_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_classifier_defect_type ON public.classifier_results USING btree (defect_type);


--
-- Name: idx_classifier_model_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_classifier_model_version ON public.classifier_results USING btree (model_version);


--
-- Name: idx_detector_model_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_detector_model_version ON public.detector_results USING btree (model_version);


--
-- Name: idx_inference_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_inference_created_at ON public.inference_requests USING btree (created_at);


--
-- Name: idx_inference_prediction; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_inference_prediction ON public.inference_requests USING btree (prediction);


--
-- Name: classifier_results classifier_results_request_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classifier_results
    ADD CONSTRAINT classifier_results_request_id_fkey FOREIGN KEY (request_id) REFERENCES public.inference_requests(id) ON DELETE CASCADE;


--
-- Name: detector_results detector_results_request_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.detector_results
    ADD CONSTRAINT detector_results_request_id_fkey FOREIGN KEY (request_id) REFERENCES public.inference_requests(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict iSpkLFVLvrY2QQSn8o2C6fiwhFTpaCNipwt9EVGNsTnPduMEIJyD4bjXexdLl46

