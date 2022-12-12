.text
.align 5

.global kernel_3

kernel_3:
    
.loop1:
    fmla v0.2d, v0.2d, v0.2d
    fmla v1.2d, v1.2d, v1.2d
    fmla v2.2d, v2.2d, v2.2d
    fmla v3.2d, v3.2d, v3.2d

    fmla v4.2d, v4.2d, v4.2d
    fmla v5.2d, v5.2d, v5.2d
    fmla v6.2d, v6.2d, v6.2d
    fmla v7.2d, v7.2d, v7.2d

    fmla v8.2d, v8.2d, v8.2d
    fmla v9.2d, v9.2d, v9.2d
    fmla v10.2d, v10.2d, v10.2d
    fmla v11.2d, v11.2d, v11.2d

    fmla v12.2d, v12.2d, v12.2d
    fmla v13.2d, v13.2d, v13.2d
    fmla v14.2d, v14.2d, v14.2d
    fmla v15.2d, v15.2d, v15.2d

    fmla v16.2d, v16.2d, v16.2d
    fmla v17.2d, v17.2d, v17.2d
    fmla v18.2d, v18.2d, v18.2d
    fmla v19.2d, v19.2d, v19.2d

    fmla v20.2d, v20.2d, v20.2d
    fmla v21.2d, v21.2d, v21.2d
    fmla v22.2d, v22.2d, v22.2d
    fmla v23.2d, v23.2d, v23.2d

    fmla v24.2d, v24.2d, v24.2d
    fmla v25.2d, v25.2d, v25.2d
    fmla v26.2d, v26.2d, v26.2d
    fmla v27.2d, v27.2d, v27.2d

    fmla v28.2d, v28.2d, v28.2d
    fmla v29.2d, v29.2d, v29.2d
    fmla v30.2d, v30.2d, v30.2d
    fmla v31.2d, v31.2d, v31.2d

    subs x0, x0, #1
    bne .loop1
    ret
