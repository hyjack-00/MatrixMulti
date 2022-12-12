.text
.align 5

.global kernel_4

kernel_4:
    
.loop1:
    mla v0.4s, v0.4s, v0.4s
    mla v1.4s, v1.4s, v1.4s
    mla v2.4s, v2.4s, v2.4s
    mla v3.4s, v3.4s, v3.4s

    mla v4.4s, v4.4s, v4.4s
    mla v5.4s, v5.4s, v5.4s
    mla v6.4s, v6.4s, v6.4s
    mla v7.4s, v7.4s, v7.4s

    mla v8.4s, v8.4s, v8.4s
    mla v9.4s, v9.4s, v9.4s
    mla v10.4s, v10.4s, v10.4s
    mla v11.4s, v11.4s, v11.4s

    mla v12.4s, v12.4s, v12.4s
    mla v13.4s, v13.4s, v13.4s
    mla v14.4s, v14.4s, v14.4s
    mla v15.4s, v15.4s, v15.4s

    mla v16.4s, v16.4s, v16.4s
    mla v17.4s, v17.4s, v17.4s
    mla v18.4s, v18.4s, v18.4s
    mla v19.4s, v19.4s, v19.4s

    mla v20.4s, v20.4s, v20.4s
    mla v21.4s, v21.4s, v21.4s
    mla v22.4s, v22.4s, v22.4s
    mla v23.4s, v23.4s, v23.4s

    mla v24.4s, v24.4s, v24.4s
    mla v25.4s, v25.4s, v25.4s
    mla v26.4s, v26.4s, v26.4s
    mla v27.4s, v27.4s, v27.4s

    mla v28.4s, v28.4s, v28.4s
    mla v29.4s, v29.4s, v29.4s
    mla v30.4s, v30.4s, v30.4s
    mla v31.4s, v31.4s, v31.4s

    subs x0, x0, #1
    bne .loop1
    ret
