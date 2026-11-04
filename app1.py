                    disabled=not gemini_available,
                    help="Local checks always run. Gemini is optional."
                )
                if not gemini_available:
                    st.caption("Gemini unavailable — local checker is ready.")

                if st.session_state.get("coherence_status") == "PENDING":
                    if st.button("Run Coherence Check", type="secondary"):
                        with st.spinner("Running coherence checks..."):
                            t0 = time.time()
                            verdict, reason, p2_confidence, p2_metrics = verify_semantic_coherence(
                                st.session_state.current_report,
                                use_gemini=use_gemini_phase2
                            )
                            t1 = time.time()
                            st.session_state.update({
                                "coherence_status": verdict,
                                "coherence_reason": reason,
                                "coherence_confidence": p2_confidence,
                                "coherence_details": p2_metrics,
                                "phase2_ms": int((t1 - t0) * 1000)
                            })
                            create_session(
                                st.session_state.session_id,
                                st.session_state.user["email"] if st.session_state.user else "anonymous",
                                st.session_state.current_report,
                                st.session_state.prediction,
                                st.session_state.confidence,
                                verdict, reason,
                                st.session_state.phase1_ms,
                                st.session_state.phase2_ms
                            )
                            st.rerun()
                elif st.session_state.coherence_status == "CONTRADICTORY":
                    st.error(f"Flagged: {st.session_state.coherence_reason}")
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")
                elif st.session_state.coherence_status == "UNVERIFIED":
                    st.warning(f"Manual review required: {st.session_state.coherence_reason}")
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")
                else:
                    st.success(st.session_state.coherence_reason)
                    st.metric("Coherence confidence", f"{st.session_state.get('coherence_confidence', 0)}%")

                if st.session_state.get("coherence_details"):
                    with st.expander("View local coherence metrics"):
                        details = st.session_state.coherence_details
                        st.metric("Software relevance", f"{details.get('relevance', 0)}%")
                        sim = details.get("similarity")
                        st.metric("Sentence similarity", "Not available" if sim is None else f"{sim}%")
                        st.caption("Matched terms: " + (", ".join(details.get("matched_terms", [])) or "None"))

            with col3:
                st.caption("Phase 3: External Routing")
                if st.session_state.user:
                    target_repo = st.text_input("Target Repo", placeholder="owner/repo")
                    custom_title = st.text_input("Issue Title", value=f"[BugTriage] {st.session_state.current_report.splitlines()[0][:65]}")
                    latest_ai = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), f"#### Raw Report\n{st.session_state.current_report}")
                    custom_desc = st.text_area("Issue Description", value=latest_ai, height=100)
                    
                    if st.button("Export to GitHub", type="primary"):
                        with st.spinner("Exporting..."):
                            success, msg = export_to_external_system(custom_title, custom_desc, st.session_state.prediction, st.session_state.confidence, st.session_state.coherence_status, target_repo, st.session_state.user["access_token"], st.session_state.user)
                            if success: st.success(msg)
                            else: st.error(msg)
                else: st.info("🔒 Sign in to export.")

        st.markdown("### Triage Assistant")
        if st.button("Standardize & Route via AI"):
            msg = "Standardize this bug report. Identify Severity, Component Routing, and Root Cause."
            st.session_state.messages.append({"role": "user", "content": msg})
            save_message(st.session_state.session_id, "user", msg)
            
            resp = generate_ai_response(msg, st.session_state.current_report, st.session_state.prediction, stream=True)
            if isinstance(resp, str): 
                st.session_state.messages.append({"role": "assistant", "content": resp})
            else:
                def stream_gen():
                    full = ""
                    for chunk in resp:
                        if chunk.text:
                            full += chunk.text
                            yield chunk.text
                    save_message(st.session_state.session_id, "assistant", full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                st.write_stream(stream_gen)
            st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

with tab_dashboard:
    st.header("System Analytics & Research Data")
    
    # 1. Lock the dashboard behind authentication
    if st.session_state.user is None:
        st.warning("🔒 Please sign in with GitHub to view your analytics and performance logs.")
    elif db is not None:
        user_email = st.session_state.user["email"]
        
        # 2. Scope all database queries strictly to the logged-in user
        total = db.sessions.count_documents({"user_email": user_email})
        
        if total > 0:
            v_count = db.sessions.count_documents({"user_email": user_email, "prediction": "Valid"})
            
            # Use $match to ensure the average is only calculated from this user's data
            agg_pipeline = [
                {"$match": {"user_email": user_email}},
                {"$group": {"_id": None, "avg": {"$avg": "$phase1_ms"}}}
            ]
            avg_p1_cursor = list(db.sessions.aggregate(agg_pipeline))
            avg_p1_result = avg_p1_cursor[0]["avg"] if avg_p1_cursor else None
            avg_p1_display = round(avg_p1_result, 2) if avg_p1_result is not None else 0
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Your Reports Processed", total)
            m_col2.metric("Valid Structure Rate", f"{round((v_count/total)*100, 1)}%")
            m_col3.metric("Avg Phase 1 Speed", f"{avg_p1_display} ms")
            
            st.subheader("Performance Logs (Research Output)")
            logs_df = pd.DataFrame(list(db.sessions.find(
                {"user_email": user_email}, 
                {"session_id":1, "prediction":1, "phase1_ms":1, "phase2_ms":1, "timestamp":1, "_id":0}
            )))
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No data available yet. Process some defect reports to generate analytics.")
    else:
